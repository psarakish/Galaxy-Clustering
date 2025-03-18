import plotly.graph_objects as go

avg_percentage, mapping_dict = scoring_metric(encoded_labels, labels_pred, return_match=True)

outlier_mapped_labels = []
for key, value in mapping_dict.items():
    if value == -1:
        print(f"True label {key} is mostly predicted as noise/outliers.")
        outlier_mapped_labels.append(key)
    else:
        print(f"True label {key} is mostly predicted as {value}")

topclusters['PredLabels'] = topclusters.index.map(lambda idx: labels_pred[idx])

true_labels_subset = topclusters['EncodedLabels'].values
pred_labels_subset = labels_pred[topclusters.index]
df_labels = pd.DataFrame({
    'TrueLabel': true_labels_subset,
    'PredLabel': pred_labels_subset
})

# Find the most common predicted label for each true label
most_common_pred_for_true = df_labels.groupby('TrueLabel')['PredLabel'].apply(lambda x: x.mode()[0])

# False positives
def is_false_positive(row):
    return row['PredLabel'] != most_common_pred_for_true[row['TrueLabel']]

misclassified_idxs = df_labels[df_labels.apply(is_false_positive, axis=1)].index.tolist()

print(f"Total false positives: {len(misclassified_idxs)}")

# Find the most common true label for each predicted label
most_common_true_for_pred = df_labels.groupby('PredLabel')['TrueLabel'].apply(lambda x: x.mode()[0])

# Find false negatives
def is_false_negative(row):
    common_pred_for_true = most_common_pred_for_true[row['TrueLabel']]
    common_true_for_pred = most_common_true_for_pred[row['PredLabel']]
    
    return (row['TrueLabel'] != common_true_for_pred) and (row['PredLabel'] == common_pred_for_true)

false_negative_idxs = df_labels[df_labels.apply(is_false_negative, axis=1)].index.tolist()

print(f"Total false negatives: {len(false_negative_idxs)}")

# Get data for top clusters
topclusters = data[data['EncodedLabels'].isin(topclusters_values)]
topclusters = topclusters.reset_index(drop=True)

# Generate a colormap for unique true labels
# Extract unique labels from both 'EncodedLabels' and 'PredLabels'
all_unique_labels = set(topclusters['EncodedLabels']).union(set(topclusters['PredLabels']))

# Create a colormap for all unique labels
colors = px.colors.qualitative.Set3
colormap = {label: colors[i % len(colors)] for i, label in enumerate(all_unique_labels)}


# Create a 3D scatter plot for the true labels
scatter_true = go.Scatter3d(
    x=topclusters['RA'],
    y=topclusters['DEC'],
    z=topclusters['Redshift'],
    mode='markers',
    marker=dict(size=5, color=topclusters['EncodedLabels'].map(colormap), opacity=0.8),
    name="True Labels"
)

# Create a 3D scatter plot for the predicted labels as empty squares
scatter_pred = go.Scatter3d(
    x=topclusters['RA'],
    y=topclusters['DEC'],
    z=topclusters['Redshift'],
    mode='markers',
    marker=dict(symbol='diamond-open', size=8, color=topclusters['PredLabels'].map(colormap), opacity=0.6),
    name="Predicted Labels"
)

# Filter data for misclassified points
misclassified_data = topclusters.loc[misclassified_idxs]
scatter_misclassified = go.Scatter3d(
    x=misclassified_data['RA'],
    y=misclassified_data['DEC'],
    z=misclassified_data['Redshift'],
    mode='markers',
    marker=dict(symbol='x', size=6, color='black', opacity=1),
    text=["x"] * len(misclassified_data),
    name="Misclassified Points"
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(nticks=10, title='RA', range=[245, 249]),
        yaxis=dict(nticks=10, title='DEC', range=[37.5, 42.5]),
        zaxis=dict(nticks=10, title='Redshift', range=[0.017, 0.05])
    )
)

fig = go.Figure(data=[scatter_true], layout=layout) #, scatter_pred, scatter_misclassified], layout=layout)
fig.show()

