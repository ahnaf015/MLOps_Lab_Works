# Kibana Dashboard Setup

After Logstash has ingested the training logs into Elasticsearch, follow these
steps to build a dashboard that visualizes ML training metrics.

## Step 1: Create a Data View (Index Pattern)

1. Open Kibana at **http://localhost:5601**
2. Log in with your `elastic` user credentials
3. Go to **Stack Management** (gear icon in the left sidebar) > **Data Views**
4. Click **Create data view**
5. Set the fields:
   - **Name:** `ML Training Metrics`
   - **Index pattern:** `ml-training-metrics-*`
   - **Timestamp field:** `@timestamp`
6. Click **Save data view to Kibana**

## Step 2: Verify Data is Flowing

1. Go to **Discover** (compass icon in the left sidebar)
2. Select the `ML Training Metrics` data view
3. Set the time range to **Last 24 hours** (or whenever you ran the pipeline)
4. You should see JSON log events. Expand one to verify fields like
   `data.model_name`, `data.accuracy`, `data.f1_score` are present

## Step 3: Create Visualizations

Go to **Dashboards** > **Create dashboard** > **Create visualization** for each:

### 3a. Model Comparison - Bar Chart

- **Type:** Bar (vertical)
- **Horizontal axis:** Terms aggregation on `data.model_name.keyword`
- **Vertical axis:** Max of `data.f1_score`
- Add layers for `data.accuracy`, `data.roc_auc`
- **Title:** "Model Performance Comparison"

### 3b. Confusion Matrix Metrics - Table

- **Type:** Lens > Table
- Filter: `event_type.keyword : "model_evaluation"`
- **Columns:** `data.model_name`, `data.true_positives`, `data.false_positives`,
  `data.true_negatives`, `data.false_negatives`
- **Title:** "Confusion Matrix Breakdown"

### 3c. False Positive vs False Negative Rate - Bar Chart

- **Type:** Bar (horizontal)
- **Vertical axis:** Terms on `data.model_name.keyword`
- **Horizontal axis:** Max of `data.false_positive_rate` and `data.false_negative_rate`
- **Title:** "Error Rate Comparison (FPR vs FNR)"

### 3d. Training Time - Bar Chart

- **Type:** Bar (vertical)
- **Horizontal axis:** Terms on `data.model_name.keyword`
- **Vertical axis:** Max of `data.training_time_seconds`
- **Title:** "Training Time by Model (seconds)"

### 3e. Dataset Profile - Metric Tiles

- **Type:** Metric
- Filter: `event_type.keyword : "data_profile"`
- Show `data.total_samples`, `data.normal_traffic`, `data.attack_traffic`
- **Title:** "Dataset Overview"

### 3f. Pipeline Events Timeline

- **Type:** Bar (vertical, stacked)
- **Horizontal axis:** Date histogram on `@timestamp`
- **Break down by:** `event_type.keyword`
- **Title:** "Pipeline Event Timeline"

## Step 4: Arrange the Dashboard

1. Drag and resize the visualizations into a clean layout:
   - Top row: Dataset Overview (metric tiles)
   - Middle row: Model Performance Comparison + Error Rate Comparison
   - Bottom row: Confusion Matrix Table + Training Time
   - Full width at bottom: Pipeline Event Timeline
2. Click **Save** and name it: `ML Training Pipeline Dashboard`

## Step 5: Take Screenshots

After the dashboard is set up:

1. Use the **Share** button (top right) > **PNG report** or take manual screenshots
2. Save screenshots in this `dashboards/` folder as:
   - `dashboard_overview.png`
   - `model_comparison.png`
   - `confusion_matrix.png`
3. These are referenced in the main README

## Tips

- If fields like `data.accuracy` show as text instead of numbers, check that
  Logstash `mutate convert` is working. You may need to refresh the data view
  field list in Stack Management.
- Use **KQL** filters in the search bar, e.g.: `event_type : "model_evaluation"` to
  focus on evaluation results only.
