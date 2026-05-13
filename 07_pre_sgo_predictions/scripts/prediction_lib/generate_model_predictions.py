import logging
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from . import utils, config as lib_config, metrics, single_review_prediction
from utils.wandb_utils import log_metrics, log_summary, get_artifact_dir, log_artifact, link_to_registry
from schemas.learned_artifacts import InitialPredictionsArtifact, InitialPredictionsArtifactLogprobs

class PipelineOrchestrator:
    def __init__(self, run, client, rate_limiter, prompt_template, config_params, category_mapping=None):
        self.run = run
        self.client = client
        self.rate_limiter = rate_limiter
        self.prompt_template = prompt_template
        # Config params include: model_name, max_tokens, temperature, num_workers, etc.
        self.params = config_params 
        # category_mapping can be full structure or just category_to_main_mapping
        # Store full structure if available, extract category_to_main_mapping for backward compatibility
        if category_mapping and isinstance(category_mapping, dict):
            if "category_to_main_mapping" in category_mapping:
                # Full structure
                self.full_category_mapping = category_mapping
                self.category_mapping = category_mapping.get("category_to_main_mapping", {})
                self.main_categories = category_mapping.get("main_categories", {})
            else:
                # Legacy format (just the mapping dict)
                self.full_category_mapping = {}
                self.category_mapping = category_mapping
                self.main_categories = {}
        else:
            self.full_category_mapping = {}
            self.category_mapping = {}
            self.main_categories = {}

    def run_pipeline(self, dataset_type, tribe_seed_data, user_backstories, review_data, 
                     topic_universe, user_to_tribe_map, output_artifact_name, target_clusters=None):
        """
        Main execution method.
        dataset_type: 'train' or 'test'
        """
        logging.info(f"Starting pipeline for {dataset_type.upper()} set...")
        
        # 1. Setup Theme Map
        category_themes_map = self._build_theme_map(topic_universe)
        logging.info(f"Built theme map with {len(category_themes_map)} categories")
        if category_themes_map:
            sample_cats = list(category_themes_map.keys())[:3]
            for cat in sample_cats:
                logging.debug(f"  Category '{cat}' has {len(category_themes_map[cat])} themes: {category_themes_map[cat][:3]}...")
        
        
        # 2. Filter Tribes if target_clusters provided
        if target_clusters:
            tribe_seed_data = self._filter_tribes(tribe_seed_data, target_clusters)

        # 3. Prepare Output Directory
        # Get output stage from config_params (defaults to "07_sgo_training" for backward compatibility)
        output_stage = self.params.get("output_stage", "07_sgo_training")
        artifact_dir = get_artifact_dir(output_stage, output_artifact_name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        all_tribe_predictions = {}
        clusters_dict = defaultdict(lambda: defaultdict(dict))

        # 4. Main Loop
        for tribe_id, tribe_seed in tribe_seed_data.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing Tribe: {tribe_id}")
            logging.info(f"{'='*60}")
            
            # Identify Users
            tribe_users = [uid for uid, tid in user_to_tribe_map.items() if tid == tribe_id]
            if not tribe_users:
                logging.warning(f"  ⚠️  No users found for tribe {tribe_id}, skipping")
                continue
            logging.info(f"  Found {len(tribe_users)} users in tribe")

            # Prepare Data for Workers
            all_review_tasks = []
            user_review_indices = defaultdict(list)
            # Track task keys to prevent duplicates (user_id + review identifier)
            task_keys_seen = set()
            
            persona_context = {
                'persona_name': tribe_seed.persona_name,
                'qualitative_summary': tribe_seed.qualitative_summary.model_dump() if hasattr(tribe_seed.qualitative_summary, 'model_dump') else tribe_seed.qualitative_summary
            }

            # Build Tasks
            users_with_reviews = 0
            reviews_skipped_no_data = 0
            reviews_skipped_no_themes = 0
            reviews_skipped_empty_themes = 0  # Reviews with empty predicted_themes
            reviews_skipped_duplicate = 0
            
            for user_id in tribe_users:
                if user_id not in review_data:
                    logging.debug(f"    User {user_id} not in review_data, skipping")
                    continue
                
                # Get User Context
                user_story = user_backstories.get(user_id)
                user_summary = user_story.overall_characteristics.influencing_characteristics_summary if user_story else ""
                if not user_summary:
                    logging.debug(f"    User {user_id} has no user summary, skipping")
                    continue
                
                reviews = review_data[user_id].get('reviews', [])
                if not reviews:
                    logging.debug(f"    User {user_id} has no reviews, skipping")
                    continue
                
                users_with_reviews += 1
                for review in reviews:
                    cat = review.get('category')
                    if not cat:
                        reviews_skipped_no_data += 1
                        continue
                    
                    # Skip reviews with empty predicted_themes
                    review_themes = review.get('themes', []) or review.get('predicted_themes', [])
                    if not review_themes or (isinstance(review_themes, list) and len(review_themes) == 0):
                        reviews_skipped_empty_themes += 1
                        logging.debug(f"    Review for user {user_id} has empty predicted_themes (skipping)")
                        continue
                    
                    # Create unique task key to prevent duplicates (user_id + asin + timestamp or review_text hash)
                    asin = review.get('asin', '')
                    timestamp = review.get('timestamp', '')
                    review_text = review.get('review_text', '')
                    # Use a combination of identifiers to create unique key
                    task_key = f"{user_id}_{asin}_{timestamp}_{hash(review_text[:50])}"
                    if task_key in task_keys_seen:
                        reviews_skipped_duplicate += 1
                        logging.debug(f"    Skipping duplicate review for user {user_id}: {asin}")
                        continue
                    task_keys_seen.add(task_key)
                    
                    main_cat = lib_config.map_category_to_main_category(cat, self.category_mapping) or cat
                    normalized_main = self._normalize_category(main_cat)
                    
                    # Always use main category (normalized) to look up themes
                    themes = category_themes_map.get(normalized_main, [])
                    
                    if not themes:
                        reviews_skipped_no_themes += 1
                        # Check if review has themes/predicted_themes that we could use
                        review_themes = review.get('themes', []) or review.get('predicted_themes', [])
                        if review_themes:
                            logging.warning(f"    Review for category '{cat}' (main: '{main_cat}') has no themes in topic_universe, but review has {len(review_themes)} themes. Available categories in topic_universe: {list(category_themes_map.keys())[:5]}")
                        else:
                            logging.debug(f"    Review for category '{cat}' (main: '{main_cat}') has no themes in topic_universe and no themes in review. Available categories: {list(category_themes_map.keys())[:5]}")
                        continue

                    # Get Category Context
                    cat_summary = ""
                    if user_story and user_story.category_characteristics:
                        c_chars = user_story.category_characteristics.get(main_cat) or user_story.category_characteristics.get(cat)
                        if c_chars: cat_summary = c_chars.influencing_characteristics_summary

                    task = (review, persona_context, user_summary, cat_summary, main_cat, themes, 
                            self.client, self.rate_limiter, self.prompt_template, 
                            self.params.get('model'), 
                            self.params.get('max_tokens'), 
                            self.params.get('temperature'),
                            self.params.get('max_retries', 3),
                            self.params.get('scoring_mode', 'confidence'),
                            self.params.get('theme_prediction_model'))
                    
                    user_review_indices[user_id].append(len(all_review_tasks))
                    all_review_tasks.append(task)

            if not all_review_tasks:
                logging.warning(f"  ⚠️  No review tasks created for tribe {tribe_id}")
                logging.warning(f"      - Users with reviews: {users_with_reviews}/{len(tribe_users)}")
                logging.warning(f"      - Reviews skipped (no data): {reviews_skipped_no_data}")
                logging.warning(f"      - Reviews skipped (empty predicted_themes): {reviews_skipped_empty_themes}")
                logging.warning(f"      - Reviews skipped (no themes in topic_universe): {reviews_skipped_no_themes}")
                logging.warning(f"      - Reviews skipped (duplicates): {reviews_skipped_duplicate}")
                continue
            
            # Log empty themes count for this tribe
            if reviews_skipped_empty_themes > 0:
                logging.warning(f"  ⚠️  Tribe {tribe_id}: {reviews_skipped_empty_themes} reviews skipped due to empty predicted_themes")
            
            if reviews_skipped_duplicate > 0:
                logging.info(f"  Skipped {reviews_skipped_duplicate} duplicate reviews (already in task list)")
            
            logging.info(f"  Created {len(all_review_tasks)} review tasks from {users_with_reviews} users")

            # Parse tribe_id to get cluster and micro IDs (needed for saving)
            cluster_id, micro_id = self._parse_tribe_id(tribe_id)

            # Run Parallel Execution
            tribe_user_predictions, tribe_failed_predictions = self._execute_parallel(all_review_tasks, user_review_indices)
            
            # Log failed predictions count
            total_failed = sum(len(failed) for failed in tribe_failed_predictions.values())
            if total_failed > 0:
                logging.warning(f"⚠️  {total_failed} failed predictions for tribe {tribe_id} (will be saved separately)")
            
            # Store only successful results
            all_tribe_predictions[tribe_id] = tribe_user_predictions
            
            # Save failed predictions separately
            if tribe_failed_predictions:
                self._save_failed_predictions(artifact_dir, cluster_id, micro_id, tribe_id, tribe_failed_predictions, dataset_type)
            
            # Organize for File Saving
            clusters_dict[cluster_id][micro_id] = tribe_user_predictions
            
            # Save Micro-Cluster Files immediately
            self._save_micro_cluster_data(
                artifact_dir, cluster_id, micro_id, tribe_id, 
                tribe_user_predictions, persona_context, dataset_type
            )
            
            # Update Cluster Grand Summary
            self._update_grand_summary(artifact_dir, cluster_id, micro_id, dataset_type)

        # 5. Verify all files are saved
        self._verify_saved_files(artifact_dir, clusters_dict)
        
        # 6. Final Logging & Artifacts
        self._log_final_artifacts(self.run, artifact_dir, output_artifact_name, all_tribe_predictions, dataset_type)

    def _execute_parallel(self, tasks, indices_map):
        """Runs single review predictions in parallel threads. Returns successful results and failed results separately.
        Thread-safe: Uses locks to prevent race conditions when collecting results from multiple threads."""
        results = defaultdict(list)
        failed_results = defaultdict(list)
        results_lock = Lock()  # Lock for thread-safe result collection
        
        with ThreadPoolExecutor(max_workers=self.params['num_workers']) as executor:
            # Map future to user_id and task index
            future_map = {}
            for uid, idxs in indices_map.items():
                for i in idxs:
                    future = executor.submit(single_review_prediction.process_single_review, tasks[i])
                    future_map[future] = (uid, i)
            
            for future in as_completed(future_map):
                uid, task_idx = future_map[future]
                try:
                    res = future.result()
                    if res:
                        # Thread-safe result collection
                        with results_lock:
                            if res.get('status') == 'failed':
                                # Separate failed predictions
                                failed_results[uid].append(res)
                            else:
                                # Successful predictions
                                results[uid].append(res)
                except Exception as e:
                    logging.error(f"Review processing exception for {uid}: {e}")
                    # Save exception as failed (thread-safe)
                    actual_review = tasks[task_idx][0]  # First element is the review
                    with results_lock:
                        failed_results[uid].append({
                            'status': 'failed',
                            'error': f'Processing exception: {str(e)}',
                            'error_type': 'exception',
                            'product_description': actual_review.get('product_description', ''),
                            'category': actual_review.get('category'),
                            'asin': actual_review.get('asin'),
                            'timestamp': actual_review.get('timestamp'),
                            'actual': {
                                'review_text': actual_review.get('review_text', ''),
                                'rating': actual_review.get('rating'),
                                'sentiment': actual_review.get('sentiment'),
                                'predicted_themes': actual_review.get('themes', actual_review.get('predicted_themes', []))
                            }
                        })
        
        return results, failed_results

    def _save_micro_cluster_data(self, base_dir, cluster_id, micro_id, tribe_id, predictions, persona_context, dataset_type):
        """Saves the individual micro-cluster summary file."""
        cluster_dir = base_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate Aggregates
        all_metrics = [r['metrics'] for u_reviews in predictions.values() for r in u_reviews if 'metrics' in r]
        
        # Helper for quant summary
        all_actuals = [{'rating': r['actual']['rating'], 'sentiment': r['actual']['sentiment']} 
                      for u_reviews in predictions.values() for r in u_reviews if 'actual' in r]
        
        avg_rating = "N/A"
        if all_actuals:
            ratings = [r['rating'] for r in all_actuals if r['rating'] is not None]
            if ratings: avg_rating = f"{np.mean(ratings):.2f}"
        
        # Calculate sentiment distribution
        sentiment_counts = Counter([r.get('sentiment', 'Unknown') for r in all_actuals if r.get('sentiment')])
        sentiment_dist = {key: round((count / len(all_actuals)) * 100, 1) for key, count in sentiment_counts.items()} if all_actuals else {}
        
        # Create aggregate_scores (raw values arrays) and final_metrics (statistics)
        aggregate_scores_raw = {}
        final_metrics = {}
        
        if all_metrics:
            # Collect all metric values into arrays
            metric_values_dict = defaultdict(list)
            for m in all_metrics:
                for key, value in m.items():
                    if key not in ['weights_used'] and isinstance(value, (int, float)):
                        metric_values_dict[key].append(value)
            
            # Create aggregate_scores (raw arrays) and final_metrics (statistics)
            for metric_name, values in metric_values_dict.items():
                if values:
                    aggregate_scores_raw[metric_name] = values
                    final_metrics[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
        else:
            # Use aggregate_metrics if we don't have raw values
            final_metrics = metrics.aggregate_metrics(all_metrics)
            
        summary_data = {
            "metadata": {
                "persona_name": persona_context.get('persona_name'),
                "micro_cluster_id": tribe_id,
                "total_users_in_cluster": len(predictions),
                "total_reviews_from_cluster": len(all_actuals),
                "quantitative_summary": {
                    "average_rating": avg_rating,
                    "sentiment_distribution_percent": sentiment_dist
                },
                "qualitative_summary": persona_context.get('qualitative_summary', {}),
                "dataset_type": dataset_type
            },
            "model_type_used": "enhanced_persona_micro_cluster",
            "user_predictions": predictions,
            "aggregate_scores": aggregate_scores_raw,  # Raw arrays of metric values
            "final_metrics": final_metrics  # Aggregated statistics (mean, std, count)
        }
        
        # Validate against schema before saving
        # Use logprobs schema if scoring_mode is "logprobs", otherwise use default schema
        scoring_mode = self.params.get('scoring_mode', 'confidence')
        schema_class = InitialPredictionsArtifactLogprobs if scoring_mode == "logprobs" else InitialPredictionsArtifact
        
        print(f"  🔍 [SCHEMA VALIDATION] Validating schema for {cluster_id}/{micro_id} (mode: {scoring_mode})...")
        logging.info(f"  🔍 Validating schema for {cluster_id}/{micro_id} (mode: {scoring_mode})...")
        try:
            validated_artifact = schema_class.from_dict(summary_data)
            validated_data = validated_artifact.to_dict()
            schema_validated = True
            print(f"  ✅ [SCHEMA VALIDATION] Schema validation PASSED for {cluster_id}/{micro_id}")
            logging.info(f"  ✅ Schema validation passed for {cluster_id}/{micro_id}")
        except Exception as e:
            print(f"  ❌ [SCHEMA VALIDATION] Schema validation FAILED for {cluster_id}/{micro_id}: {e}")
            logging.error(f"  ❌ Schema validation failed for {cluster_id}/{micro_id}: {e}")
            logging.error(f"  ❌ Validation error details: {str(e)}")
            logging.error("  Saving without schema validation (data may be invalid)")
            validated_data = utils.convert_to_serializable(summary_data)
            schema_validated = False
        
        # Naming Convention Difference
        infix = "_test" if dataset_type == "test" else ""
        filename = f"{micro_id}{infix}_summary_enhanced_persona_micro_cluster_accuracy.json"
        
        with open(cluster_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, indent=2, ensure_ascii=False)
        
        if schema_validated:
            logging.info(f"  💾 Saved {cluster_id}/{filename}: {len(predictions)} users, {len(all_actuals)} reviews with metrics (schema validated)")
        else:
            logging.warning(f"  ⚠️  Saved {cluster_id}/{filename}: {len(predictions)} users, {len(all_actuals)} reviews with metrics (schema validation failed)")

    def _save_failed_predictions(self, base_dir, cluster_id, micro_id, tribe_id, failed_predictions, dataset_type):
        """Saves failed predictions to a separate file for reprocessing."""
        cluster_dir = base_dir / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        # Naming convention for failed predictions file
        infix = "_test" if dataset_type == "test" else ""
        failed_filename = f"{micro_id}{infix}_failed_predictions.json"
        failed_file = cluster_dir / failed_filename
        
        # Prepare failed predictions data with metadata
        failed_data = {
            "metadata": {
                "micro_cluster_id": tribe_id,
                "cluster_id": cluster_id,
                "micro_id": micro_id,
                "total_failed_reviews": sum(len(reviews) for reviews in failed_predictions.values()),
                "total_failed_users": len(failed_predictions),
                "dataset_type": dataset_type,
                "note": "These predictions failed after all retries and can be reprocessed later"
            },
            "failed_predictions": failed_predictions  # Format: {user_id: [list of failed reviews]}
        }
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(utils.convert_to_serializable(failed_data), f, indent=2, ensure_ascii=False)
        
        total_failed = sum(len(reviews) for reviews in failed_predictions.values())
        logging.warning(f"  ⚠️  Saved {cluster_id}/{failed_filename}: {len(failed_predictions)} users, {total_failed} failed reviews")

    def _update_grand_summary(self, base_dir, cluster_id, micro_id, dataset_type):
        """Updates the cluster-level summary."""
        cluster_dir = base_dir / cluster_id
        infix = "_test" if dataset_type == "test" else ""
        grand_filename = f"grand_summary{infix}_enhanced_persona_micro_cluster.json"
        micro_pattern = f"micro_*{infix}_summary_enhanced_persona_micro_cluster_accuracy.json"
        
        all_cluster_metrics = []
        
        # Read all existing files in this cluster folder to aggregate
        for fpath in cluster_dir.glob(micro_pattern):
            try:
                data = utils.load_json_file(fpath)
                for u_reviews in data.get("user_predictions", {}).values():
                    for r in u_reviews:
                        if "metrics" in r: all_cluster_metrics.append(r["metrics"])
            except: pass
            
        grand_summary = {
            "model_type_used": "enhanced_persona_micro_cluster",
            "cluster_name": cluster_id,
            "final_summary": metrics.aggregate_metrics(all_cluster_metrics),
            "dataset_type": dataset_type
        }
        
        with open(cluster_dir / grand_filename, 'w', encoding='utf-8') as f:
            json.dump(utils.convert_to_serializable(grand_summary), f, indent=2, ensure_ascii=False)

    def _verify_saved_files(self, artifact_dir, clusters_dict):
        """Verifies all files are saved correctly."""
        logging.info(f"\n{'='*60}")
        logging.info(f"Verifying all predictions are saved...")
        logging.info(f"{'='*60}")
        
        total_clusters = len(clusters_dict)
        total_micro_clusters = sum(len(micro_clusters) for micro_clusters in clusters_dict.values())
        total_users_saved = 0
        total_reviews_saved = 0
        
        for cluster_id, micro_clusters in clusters_dict.items():
            cluster_dir = artifact_dir / cluster_id
            cluster_total_users = 0
            cluster_total_reviews = 0
            
            for micro_id, tribe_data in micro_clusters.items():
                # Check for summary file (the only file we create per micro-cluster)
                summary_file = cluster_dir / f"{micro_id}_summary_enhanced_persona_micro_cluster_accuracy.json"
                if summary_file.exists():
                    try:
                        summary_data = utils.load_json_file(summary_file)
                        if summary_data:
                            num_users = summary_data.get("metadata", {}).get("total_users_in_cluster", 0)
                            num_reviews = summary_data.get("metadata", {}).get("total_reviews_from_cluster", 0)
                            cluster_total_users += num_users
                            cluster_total_reviews += num_reviews
                    except Exception as e:
                        logging.warning(f"  ⚠️  Could not read {cluster_id}/{summary_file.name}: {e}")
                else:
                    logging.warning(f"  ⚠️  File not found: {cluster_id}/{micro_id}_summary_enhanced_persona_micro_cluster_accuracy.json")
            
            total_users_saved += cluster_total_users
            total_reviews_saved += cluster_total_reviews
            
            grand_summary_file = cluster_dir / "grand_summary_enhanced_persona_micro_cluster.json"
            if grand_summary_file.exists():
                logging.info(f"  ✅ {cluster_id}: {len(micro_clusters)} micro-clusters, {cluster_total_users} users, {cluster_total_reviews} reviews")
            else:
                logging.warning(f"  ⚠️  {cluster_id}/grand_summary_enhanced_persona_micro_cluster.json not found")
        
        logging.info(f"\n✅ Verification complete:")
        logging.info(f"   - {total_clusters} clusters")
        logging.info(f"   - {total_micro_clusters} micro-clusters")
        logging.info(f"   - {total_users_saved} users")
        logging.info(f"   - {total_reviews_saved} reviews")

    def _log_final_artifacts(self, run, artifact_dir, artifact_name, all_predictions, dataset_type):
        """Logs to W&B."""
        total_users = sum(len(users) for users in all_predictions.values())
        total_reviews = sum(sum(len(u) for u in users.values()) for users in all_predictions.values())
        
        # Count failed predictions and check schema validation across all clusters
        infix = "_test" if dataset_type == "test" else ""
        total_failed = 0
        total_failed_users = 0
        schema_validation_passed = True
        schema_validation_failed_count = 0
        
        for cluster_dir in artifact_dir.iterdir():
            if cluster_dir.is_dir():
                for failed_file in cluster_dir.glob(f"*{infix}_failed_predictions.json"):
                    try:
                        failed_data = utils.load_json_file(failed_file)
                        if failed_data:
                            total_failed += failed_data.get("metadata", {}).get("total_failed_reviews", 0)
                            total_failed_users += failed_data.get("metadata", {}).get("total_failed_users", 0)
                    except Exception as e:
                        logging.warning(f"Could not read failed predictions from {failed_file}: {e}")
                
                # Check schema validation status from summary files
                for summary_file in cluster_dir.glob(f"micro_*{infix}_summary_enhanced_persona_micro_cluster_accuracy.json"):
                    try:
                        summary_data = utils.load_json_file(summary_file)
                        if summary_data:
                            print(f"  🔍 [SCHEMA VALIDATION] Re-validating {summary_file.name}...")
                            # Use logprobs schema if scoring_mode is "logprobs"
                            scoring_mode = self.params.get('scoring_mode', 'confidence')
                            schema_class = InitialPredictionsArtifactLogprobs if scoring_mode == "logprobs" else InitialPredictionsArtifact
                            schema_class.from_dict(summary_data)
                            print(f"  ✅ [SCHEMA VALIDATION] Re-validation PASSED for {summary_file.name}")
                    except Exception as e:
                        schema_validation_passed = False
                        schema_validation_failed_count += 1
                        print(f"  ❌ [SCHEMA VALIDATION] Re-validation FAILED for {summary_file.name}: {e}")
                        logging.warning(f"Schema validation check failed for {summary_file.name}: {e}")
        
        log_metrics(run, {
            "num_tribes": len(all_predictions),
            "total_users": total_users,
            "total_reviews": total_reviews,
            "avg_reviews_per_tribe": total_reviews / len(all_predictions) if all_predictions else 0,
            "total_failed_reviews": total_failed,
            "total_failed_users": total_failed_users,
            "schema_validation_passed": schema_validation_passed,
            "schema_validation_failed_count": schema_validation_failed_count
        })
        
        if total_failed > 0:
            print(f"⚠️  [FAILED PREDICTIONS] Total failed predictions: {total_failed} reviews from {total_failed_users} users (saved separately for reprocessing)")
            logging.warning(f"⚠️  Total failed predictions: {total_failed} reviews from {total_failed_users} users (saved separately for reprocessing)")
        
        if not schema_validation_passed:
            print(f"⚠️  [SCHEMA VALIDATION] Schema validation failed for {schema_validation_failed_count} micro-cluster(s)")
            logging.warning(f"⚠️  Schema validation failed for {schema_validation_failed_count} micro-cluster(s)")
        else:
            print(f"✅ [SCHEMA VALIDATION] All micro-clusters passed schema validation!")
        
        artifact = log_artifact(
            run=run,
            artifact_name=artifact_name,
            artifact_type="dataset",
            artifact_path=artifact_dir,
            metadata={
                "dataset_type": dataset_type, 
                "total_reviews": total_reviews,
                "schema_version": "v4",
                "schema_validated": schema_validation_passed,
                "artifact_type": "learned_artifact"
            }
        )
        # Use appropriate stage for registry linking (from config_params)
        output_stage = self.params.get("output_stage", "07_sgo_training")
        link_to_registry(artifact, stage=output_stage)
        log_summary(run, {
            "status": "completed",
            "dataset_type": dataset_type,
            "tribes_processed": len(all_predictions),
            "users_processed": total_users,
            "reviews_processed": total_reviews
        })

    # --- Helpers ---
    def _normalize_category(self, category):
        """Normalize category name to use underscores: replace spaces and special chars with underscores."""
        if not category:
            return category
        # Replace & with "and" first (so "Health & Personal Care" -> "Health and Personal Care")
        # Then replace spaces and special chars with underscores
        normalized = category.replace(' & ', ' and ').replace('&', ' and ')
        normalized = normalized.replace(' ', '_').replace('-', '_').replace('/', '_')
        # Collapse multiple underscores
        normalized = '_'.join(filter(None, normalized.split('_')))  # Remove empty parts
        return normalized
    
    def _build_theme_map(self, topic_universe):
        """Build theme map from topic_universe, normalizing all category names to use underscores."""
        mapping = {}
        if isinstance(topic_universe, dict):
            for cat, data in topic_universe.items():
                if isinstance(data, list):
                    themes = [t.get('topic_name', t) if isinstance(t, dict) else str(t) for t in data]
                elif isinstance(data, dict) and 'topics' in data:
                    themes = [t.get('topic_name', t) if isinstance(t, dict) else str(t) for t in data['topics']]
                else:
                    continue
                
                # Store with normalized key (underscores only)
                normalized_key = self._normalize_category(cat)
                mapping[normalized_key] = themes
        return mapping

    def _parse_tribe_id(self, tribe_id):
        # Extracts cluster_X and micro_Y from tribe_id
        if "_micro_" in tribe_id:
            parts = tribe_id.split("_micro_")
            raw_cluster = parts[0]
            micro_id = f"micro_{parts[1]}"
            
            # Normalize segment_0 -> cluster_0
            match = re.search(r'(\d+)', raw_cluster)
            cluster_id = f"cluster_{match.group(1)}" if match else raw_cluster
            return cluster_id, micro_id
        return tribe_id, "micro_0"

    def _filter_tribes(self, tribe_data, target_clusters):
        """Filter tribes by cluster names. Accepts both 'cluster_X' and 'segment_X' formats."""
        if not target_clusters:
            return tribe_data
        
        # Normalize cluster names (accept both "cluster_0" and "segment_0" formats)
        # We'll match against the actual tribe_id format (segment_X) but save as cluster_X
        target_clusters_raw = set()
        target_clusters_segment = set()  # For matching tribe_ids
        
        for cluster_name in target_clusters:
            target_clusters_raw.add(cluster_name)
            # Convert to segment format for matching tribe_ids
            if cluster_name.startswith("cluster_"):
                segment_name = cluster_name.replace("cluster_", "segment_", 1)
                target_clusters_segment.add(segment_name)
            elif cluster_name.startswith("segment_"):
                target_clusters_segment.add(cluster_name)
            else:
                # Try to extract number and create both formats
                match = re.search(r'(\d+)', cluster_name)
                if match:
                    num = match.group(1)
                    target_clusters_segment.add(f"segment_{num}")
        
        # Filter tribes that belong to target clusters
        filtered_tribe_seed_data = {}
        for tribe_id, tribe_seed in tribe_data.items():
            # Extract cluster from tribe_id (e.g., "segment_0_micro_0" -> "segment_0")
            if "_micro_" in tribe_id:
                cluster_id = tribe_id.split("_micro_")[0]
            else:
                cluster_id = tribe_id
            
            if cluster_id in target_clusters_segment:
                filtered_tribe_seed_data[tribe_id] = tribe_seed
        
        if not filtered_tribe_seed_data:
            logging.error(f"No tribes found for specified clusters: {target_clusters}")
            available = set(tid.split('_micro_')[0] if '_micro_' in tid else tid for tid in tribe_data.keys())
            logging.error(f"Available clusters: {available}")
            return {}
        
        logging.info(f"Filtered to {len(filtered_tribe_seed_data)} tribes in clusters: {sorted(target_clusters_raw)}")
        return filtered_tribe_seed_data