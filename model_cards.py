import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import sqlite3
from openai import OpenAI
import tiktoken
import json

def query_sqlite3(query : str, db : str = 'PeaTMOSS_DIST.db') -> str | None:
    conn = None
    cur = None
    try:
        conn = sqlite3.connect(db, timeout=10)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return result
    except Exception as e:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        print(f"An error occurred: {e}")
        return None

def create_model_cards_df():
    model_cards_1_df = pd.read_parquet('model-cards-0')
    model_cards_2_df = pd.read_parquet('model-cards-1')
    model_cards_df = pd.concat([model_cards_1_df, model_cards_2_df])

    print(model_cards_df.head())
    print(model_cards_df.shape)

    # Each modelId in the model_cards_df is a unique identifier for a model. We can use this to query the model's name from the models table.
    # I want to determine the amount of modelIds that are in the models table.
    # I also want to determine the amount of models that don't have a corresponding modelId in the model_cards_df.
    model_ids = model_cards_df['modelId'].tolist()
    model_ids_str = ', '.join([f"'{model_id}'" for model_id in model_ids])
    query = f"""
        SELECT context_id
        FROM model
        WHERE context_id IN ({model_ids_str});
    """
    model_context_ids = query_sqlite3(query)
    model_context_ids = [model_context_id[0] for model_context_id in model_context_ids]
    print(len(model_context_ids))

    # Only keep the rows with a modelId that is in the model table
    model_cards_df = model_cards_df[model_cards_df['modelId'].isin(model_context_ids)]
    print(model_cards_df.shape)
    return model_cards_df

def create_model_data_df():
    query = """
    -- Obtain model data from PTM and Enhanced PTM metadata tables, including specified tables
    -- Exclude discussions, pull requests, and issues
    SELECT 
        model.context_id AS context_id,
        model.downloads,
        model.likes,
        architecture.name AS architecture_name,
        author.name AS author_name,
        dataset.name AS dataset_name,
        dataset.url AS dataset_url,
        framework.name AS framework_name,
        license.name AS license_name,
        paper.title AS paper_title,
        library.name AS library_name,
        tag.name AS tag_name,
        domain.name AS domain_name,
        model_task.name AS task_name,
        hardware.name AS hardware_name,
        grant.name AS grant_name,
        github_repo.url AS github_repo_url
    FROM
        model
    LEFT JOIN model_to_architecture ON model.id = model_to_architecture.model_id
    LEFT JOIN architecture ON model_to_architecture.architecture_id = architecture.id
    LEFT JOIN model_to_author ON model.id = model_to_author.model_id
    LEFT JOIN author ON model_to_author.author_id = author.id
    LEFT JOIN model_to_dataset ON model.id = model_to_dataset.model_id
    LEFT JOIN dataset ON model_to_dataset.dataset_id = dataset.id
    LEFT JOIN model_to_framework ON model.id = model_to_framework.model_id
    LEFT JOIN framework ON model_to_framework.framework_id = framework.id
    LEFT JOIN model_to_license ON model.id = model_to_license.model_id
    LEFT JOIN license ON model_to_license.license_id = license.id
    LEFT JOIN model_to_paper ON model.id = model_to_paper.model_id
    LEFT JOIN paper ON model_to_paper.paper_id = paper.id
    LEFT JOIN model_to_library ON model.id = model_to_library.model_id
    LEFT JOIN library ON model_to_library.library_id = library.id
    LEFT JOIN model_to_tag ON model.id = model_to_tag.model_id
    LEFT JOIN tag ON model_to_tag.tag_id = tag.id
    LEFT JOIN model_to_domain ON model.id = model_to_domain.model_id
    LEFT JOIN domain ON model_to_domain.domain_id = domain.id
    LEFT JOIN model_to_model_task ON model.id = model_to_model_task.model_id
    LEFT JOIN model_task ON model_to_model_task.model_task_id = model_task.id
    LEFT JOIN model_to_hardware ON model.id = model_to_hardware.model_id
    LEFT JOIN hardware ON model_to_hardware.hardware_id = hardware.id
    LEFT JOIN model_to_grant ON model.id = model_to_grant.model_id
    LEFT JOIN grant ON model_to_grant.grant_id = grant.id
    LEFT JOIN github_repo ON model.id = github_repo.model_id;
    """

    model_data = query_sqlite3(query)
    model_data_df = pd.DataFrame(model_data, columns=['context_id', 'downloads', 'likes', 'architecture_name', 'author_name', 'dataset_name', 'dataset_url', 'framework_name', 'license_name', 'paper_title', 'library_name', 'tag_name', 'domain_name', 'task_name', 'hardware_name', 'grant_name', 'github_repo_url'])
    # Consolidate this on context_id. For each column, I want to consolidate the data into a set.
    model_data_df = model_data_df.groupby('context_id').agg(set).reset_index()

    return model_data_df

def create_model_scores_df():
    cards_df = create_model_cards_df()
    data_df = create_model_data_df()

    # The only columns I need from cards_df is the modelId and the modelCard (card). I will merge this with data_df on the context_id.
    cards_df = cards_df[['modelId', 'card']]
    model_df = data_df.merge(cards_df, left_on='context_id', right_on='modelId')
    model_df = model_df.drop('modelId', axis=1)

    model_df = model_df.applymap(lambda x: None if x == set([None]) else x)
    model_df['metadataScore'] = model_df.applymap(lambda x: 1 if x is not None else 0).mean(axis=1)

    # Now I will calculate the final score.
    model_df['finalScore'] = (model_df['modelCardScore'] + model_df['metadataScore']) / 2

    # I will sort the models by the final score.
    model_df = model_df.sort_values('finalScore', ascending=False)

    # I will print the top 10 models.
    print(model_df.head(10))

    # I will save the model_df to a parquet file.
    model_df.to_parquet('model_scores.parquet')
    return model_df

# I will now use OpenAI's GPT model to analyze the model card.
api_key = ""
client = OpenAI(api_key=api_key)

def analyze_model_card(model_card: str) -> tuple[dict, str]:
    prompt = f"""
    You will receive a model card and are expected to analyze it for the following details:
    1. Model description: A description of the model itself
    2. Limitations: Any limitations of the model
    3. How to use: Instructions on how to use the model downstream
    4. Training: Details of the training process or data
    5. Evaluation: Reports on the model's performance evaluation

    Please respond with a JSON object indicating whether each of these points is present with true/false.
    Here is the model card to evaluate:
    {model_card}
    """

    # Generate a completion with OpenAI's GPT model
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a model card evaluator."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4"
    )

    # Extract and return the JSON object containing the analysis
    analysis = completion.choices[0].message.content.strip()


    return analysis


import re
# The column analysis_result has a JSON object as a string. I will convert this to a dictionary.
# I will then count the number of true values for each key in the dictionary, then divide by the total amount of keys.
# This will convert analysis into a dict if possible.
def calculate_model_card_score(row):
    if row['analysis_result'] is None:
        return None
    pattern = r"\{([^}]*)\}"

    # Find all matches using the regular expression pattern
    matches = re.findall(pattern, row['analysis_result'])

    try:
        analysis = json.loads(f"{{{matches[0]}}}")
    except:
        print(row['analysis_result'])
        return None

    for key, value in analysis.items():
        if type(value) is not bool and value not in ["true", "false"]:
            print(analysis)
            return None
        elif value == "true":
            analysis[key] = True
        elif value == "false":
            analysis[key] = False

    return analysis

def calculate_model_card_score(row):
    if row['analysis'] is None:
        return 0
    # Count the number of trues in the analysis dictionary and divide by 5
    # There may be None values in the dictionary. I will replace these with False.
    row['analysis'] = {key: value for key, value in row['analysis'].items() if value is not None}
    return sum(row['analysis'].values()) / 5

# Plot Box and whisker plots of the top and bottom 1k models finalScores on the same plot
def plot_final_scores(top_popular_models, bottom_popular_models, title='Final Scores of Top and Bottom 1k Models', metric='downloads'):
    fig, ax = plt.subplots()
    ax.boxplot([top_popular_models['Documentation Quality Score'], bottom_popular_models['Documentation Quality Score']], labels=[f'Top 1k by {metric}', f'Bottom 1k by {metric}'])
    ax.set_title(title)
    ax.set_ylabel('Documentation Quality Score')
    plt.show()

def plot_final_score_same_plot(top_popular_models, bottom_popular_models):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].boxplot([top_popular_models.nlargest(1000, 'downloads')['Documentation Quality Score'], bottom_popular_models.nlargest(1000, 'downloads')['Documentation Quality Score']], labels=[f'Top 1k', f'Bottom 1k'])
    ax[0].set_title('Downloads')
    ax[0].set_ylabel('Documentation Quality Score')

    ax[1].boxplot([top_popular_models.nlargest(1000, 'likes')['Documentation Quality Score'], bottom_popular_models.nlargest(1000, 'likes')['Documentation Quality Score']], labels=[f'Top 1k', f'Bottom 1k'])
    ax[1].set_title('Likes')
    ax[1].set_ylabel('Documentation Quality Score')

    ax[2].boxplot([top_popular_models.nlargest(1000, 'num_downstream_repos')['Documentation Quality Score'], bottom_popular_models.nlargest(1000, 'num_downstream_repos')['Documentation Quality Score']], labels=[f'Top 1k', f'Bottom 1k'])
    ax[2].set_title('Downstream Repositories')
    ax[2].set_ylabel('Documentation Quality Score')
    plt.show()

model_scores = create_model_scores_df()
top_popular_models = pd.read_parquet('top_1k_model_card_scores.parquet')
bottom_popular_models = pd.read_parquet('bottom_1k_model_card_scores.parquet')

# Full metadata
model_data = create_model_data_df()
print(model_data.head())
# Calculate the metadata score for each model, only considering the columns:
#  ['architecture_name', 'author_name', 'dataset_name', 'dataset_url', 'framework_name', 'license_name', 'paper_title', 'library_name', 'tag_name', 'domain_name', 'task_name', 'hardware_name', 'grant_name', 'github_repo_url']
# This will be the number of non-null values divided by the total number of columns
['architecture_name', 'author_name', 'dataset_name', 'dataset_url', 'framework_name', 'license_name', 'paper_title', 'library_name', 'tag_name', 'domain_name', 'task_name', 'hardware_name', 'grant_name', 'github_repo_url']
model_data = model_data.applymap(lambda x: None if x == set([None]) else x)
model_data['metadataScore'] = model_data.applymap(lambda x: 1 if x is not None else 0).mean(axis=1)
print(model_data.head())

# Update the metadataScore column in the top_popular_models and bottom_popular_models DataFrames based on the metadataScore column in the model_data DataFrame
for index, row in top_popular_models.iterrows():
    top_popular_models.at[index, 'metadataScore'] = model_data[model_data['context_id'] == row['context_id']]['metadataScore'].values[0]
for index, row in bottom_popular_models.iterrows():
    bottom_popular_models.at[index, 'metadataScore'] = model_data[model_data['context_id'] == row['context_id']]['metadataScore'].values[0]

top_popular_models['modelCardScore'] = top_popular_models.apply(calculate_model_card_score, axis=1)
bottom_popular_models['modelCardScore'] = bottom_popular_models.apply(calculate_model_card_score, axis=1)

top_popular_models['finalScore'] = None
bottom_popular_models['finalScore'] = None

# Only calculate the final score if the modelCardScore is not None
top_popular_models.loc[top_popular_models['modelCardScore'].notnull(), 'finalScore'] = (top_popular_models['modelCardScore'] + top_popular_models['metadataScore']) / 2
bottom_popular_models.loc[bottom_popular_models['modelCardScore'].notnull(), 'finalScore'] = (bottom_popular_models['modelCardScore'] + bottom_popular_models['metadataScore']) / 2

top_popular_models = top_popular_models.sort_values('finalScore', ascending=False)
bottom_popular_models = bottom_popular_models.sort_values('finalScore', ascending=False)

top_popular_models = pd.read_csv('top_model_card_scores.csv')
bottom_popular_models = pd.read_csv('bottom_model_card_scores.csv')

plot_final_score_same_plot(top_popular_models, bottom_popular_models)

# plot_final_scores(top_popular_models.nlargest(1000, 'downloads'), bottom_popular_models.nlargest(1000, 'downloads'), "Documentation Quality Scores of Top and Bottom 1k Models by Downloads", "Downloads")
# plot_final_scores(top_popular_models.nlargest(1000, 'likes'), bottom_popular_models.nlargest(1000, 'likes'), "Documentation Quality Scores of Top and Bottom 1k Models by Likes", "Likes")
# plot_final_scores(top_popular_models.nlargest(1000, 'num_downstream_repos'), bottom_popular_models.nlargest(1000, 'num_downstream_repos'), "Documentation Quality Scores of Top and Bottom 1k Models by Number of Downstream Repos", "Number of Downstream Repos")

# # I will now store the top_popular_models and bottom_popular_models DataFrames to parquet files.
# top_popular_models.to_parquet('top_1k_model_card_scores_complete.parquet')
# bottom_popular_models.to_parquet('bottom_1k_model_card_scores_complete.parquet')

# # I will also store just the scores and the popularity metrics used in a separate csv file.
# # The column names will be renamed to include the metric used to sort the models.
# top_popular_models_scores = top_popular_models[['context_id', 'modelCardScore', 'metadataScore', 'finalScore', 'downloads', 'likes', 'num_downstream_repos']].rename(columns={'finalScore': 'Documentation Quality Score', 'metadataScore': 'Metadata Quality Score', 'modelCardScore': 'Model Card Quality Score'})
# bottom_popular_models_scores = bottom_popular_models[['context_id', 'modelCardScore', 'metadataScore', 'finalScore', 'downloads', 'likes', 'num_downstream_repos']].rename(columns={'finalScore': 'Documentation Quality Score', 'metadataScore': 'Metadata Quality Score', 'modelCardScore': 'Model Card Quality Score'})

# top_popular_models_scores.to_csv('top_model_card_scores.csv', index=False)
# bottom_popular_models_scores.to_csv('bottom_model_card_scores.csv', index=False)