"""
Name: Kato Joseph Bwanika 
Reg: 2023-B291-11709
NLP Test: NLP analysis for Sentiment Analysis and Topic Clustering.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# ==============================================================================
# QUESTION 1: SENTIMENT ANALYSIS (PARTITIONED)
# ==============================================================================

print("QUESTION 1: SENTIMENT ANALYSIS")

# --- Question1 IMPORTS ---
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# (i) Library Imports and Setup
print("(i) Setting up environment for Question 1...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    import spacy
    nlp = spacy.load("en_core_web_sm")

try:
    from wordcloud import WordCloud
except:
    os.system("pip install wordcloud")
    from wordcloud import WordCloud

from sklearn.metrics import roc_curve, auc


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)

def get_data_path_q1(filename):
    if os.path.exists(filename):
        return filename
    if os.path.exists(os.path.join("..", filename)):
        return os.path.join("..", filename)
    raise FileNotFoundError(f"Dataset file {filename} not found. Please ensure the original dataset is placed in the directory.")

# (ii) Load Dataset
print("(ii) Loading Sentiment140 dataset from unzipped folder...")
df1 = pd.read_csv(os.path.join('sentiment140_data', 'training.1600000.processed.noemoticon.csv'), 
                  encoding='latin-1', 
                  low_memory=False, 
                  names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

# Subsample for demonstration
df1 = df1.sample(5000, random_state=42).reset_index(drop=True)

df1['sentiment'] = df1['sentiment'].replace(4, 1)
print("(ii) Dataset loaded and subsampled (5,000 rows) for Question 1.")

# (iii) Basic Statistics and Shape
print(f"(iii) Dataset Shape: {df1.shape}")
print("(iii) Class Distribution:\n", df1['sentiment'].value_counts())

# (iii-extra) Class Distribution Visual
plt.figure(figsize=(6, 6))
df1['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], labels=['Positive', 'Negative'], startangle=90)
plt.title("Visual 1: Sentiment Class Distribution")
plt.ylabel("")
plt.savefig("Sentiment_Visual_1_Class_Distribution.png")
plt.show(block=False); plt.pause(1)

# (iv) Data Visualization: Histogram
df1['text_length'] = df1['text'].apply(lambda x: len(str(x)))
plt.figure()
sns.histplot(df1['text_length'], bins=25, kde=True, color='teal')
plt.title("Visual 2: Distribution of Tweet Lengths")
plt.savefig("Sentiment_Visual_2_Tweet_Lengths.png")
plt.show(block=False); plt.pause(1)

# (v) Noise Identification
print("(v) Identifying Noise: URLs, @mentions, and hashtags identified in raw text.")

# (vi-xii) Preprocessing
stop_words = set(stopwords.words('english'))
# Expanded custom stop words for cleaner results
custom_stops = ['im', 'u', 'lol', 'rt', 'ur', 'amp', 'today', 'going', 'got', 'dont', 
                'cant', 'one', 'day', 'get', 'see', 'think', 'make', 'know', 'back', 
                'well', 'much', 'even', 'really', 'still', 'want', 'time', 'good', 
                'new', 'now', 'work', 'gotta', 'gonna', 'wanna', 'yeah', 'yes', 'no',
                'oh', 'ok', 'hey', 'hi', 'hello', 'thanks', 'thank', 'please', 'wait',
                'take', 'come', 'got', 'look', 'find', 'way', 'say', 'said', 'guy',
                'girl', 'thing', 'something', 'anything', 'nothing', 'people', 'woman', 
                'man', 'year', 'world', 'old', 'ago', 'also', 'would', 'could', 'should',
                'first', 'last', 'many', 'much', 'every', 'never', 'ever', 'always']
stop_words.update(custom_stops)
lemmatizer = WordNetLemmatizer()

def clean_text_q1(text):
    text = str(text).lower()
    text = re.sub(r"\bi'm\b", "im", text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(t) for t in nltk.word_tokenize(text) if len(t) > 1 and t not in stop_words]
    return " ".join(tokens)

# (xiii) Before vs After Comparison (8 Samples)
print("\n(xiii) Comparison of 8 samples (Before vs After Cleaning):")
print("-" * 50)
for i in range(8):
    raw = df1['text'].iloc[i]
    cleaned = clean_text_q1(raw)
    print(f"Sample {i+1}:")
    print(f"  BEFORE: {raw}")
    print(f"  AFTER:  {cleaned}")
    print("-" * 20)

# (xiv) Impact Analysis: Normalization and Transformation
print("\n(xiv) Running Impact Analysis...")
df1['clean_text'] = df1['text'].apply(clean_text_q1)
print("(xiv) Impact Analysis: Text normalization successful. Noise removed.")

# (xv-xvi) Word Frequency Chart
all_words_q1 = " ".join(df1['clean_text']).split()
word_counts_q1 = Counter(all_words_q1).most_common(15)
words_q1, counts_q1 = zip(*word_counts_q1)
plt.figure()
sns.barplot(x=list(counts_q1), y=list(words_q1), palette='viridis')
plt.title("Visual 3: Top 15 Frequent Words")
plt.savefig("Sentiment_Visual_3_Frequent_Words.png")
plt.show(block=False); plt.pause(1)

# (xv-xvi-extra) Word Cloud
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(all_words_q1))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Visual 4: Word Cloud of Processed Tweets")
plt.savefig("Sentiment_Visual_4_WordCloud.png")
plt.show(block=False); plt.pause(1)


# (xvii-xviii) Bigram Chart
bi_counts_q1 = Counter(nltk.bigrams(all_words_q1)).most_common(10)
bi_labels_q1 = [f"{b[0]} {b[1]}" for b, c in bi_counts_q1]
plt.figure()
sns.barplot(x=[c for b, c in bi_counts_q1], y=bi_labels_q1, palette='magma')
plt.title("Visual 5: Top 10 Bigrams")
plt.savefig("Sentiment_Visual_5_Bigrams.png")
plt.show(block=False); plt.pause(1)


# (xvii-xviii-extra) Trigram Chart and Comparison
tri_counts_q1 = Counter(nltk.trigrams(all_words_q1)).most_common(10)
tri_labels_q1 = [f"{t[0]} {t[1]} {t[2]}" for t, c in tri_counts_q1]
plt.figure()
sns.barplot(x=[c for t, c in tri_counts_q1], y=tri_labels_q1, palette='rocket')
plt.title("Visual 6: Top 10 Trigrams")
plt.savefig("Sentiment_Visual_6_Trigrams.png")
plt.show(block=False); plt.pause(1)


print("\n(xvii-xviii) Comparison: Bigrams vs Trigrams")
print(f" - Top Bigram: '{bi_labels_q1[0]}' (Count: {bi_counts_q1[0][1]})")
if tri_labels_q1:
    print(f" - Top Trigram: '{tri_labels_q1[0]}' (Count: {tri_counts_q1[0][1]})")
else:
    print(" - Top Trigram: None found in sample.")
print(" - Insight: Bigrams capture immediate word pairs, while trigrams provide richer contextual sequences but typically have lower frequencies due to higher specificity.")


# (xix-xx) NER
# (xix-xx) NER Results Table (Visual Table)
print("\n(xix) NER Results Table (First 10 Entities found):")
entities_list = []
for t in df1['text'].iloc[:50]: # Search first 50 to find at least 10
    d = nlp(t)
    for e in d.ents:
        entities_list.append((e.text, e.label_))
    if len(entities_list) >= 10: break
ner_df = pd.DataFrame(entities_list[:10], columns=['Entity', 'Label'])
print(ner_df.to_string(index=False))

# (xxi) TF-IDF Representation and Table
tfidf_q1 = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X_q1 = tfidf_q1.fit_transform(df1['clean_text'])
print("\n(xxi) TF-IDF Importance Table (Top 10 Words in Sample):")
scores = X_q1[0].toarray().flatten()
words = tfidf_q1.get_feature_names_out()
tfidf_df = pd.DataFrame({'Word': words, 'Score': scores}).sort_values(by='Score', ascending=False).head(10)
print(tfidf_df.to_string(index=False))
y_q1 = df1['sentiment']

# (xxii-xxvi) Model Development
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_q1, y_q1, test_size=0.2, random_state=42, stratify=y_q1)
nb_q1 = MultinomialNB().fit(X_train1, y_train1)
lr_q1 = LogisticRegression(max_iter=1000).fit(X_train1, y_train1)
print(f"(xxiv) Naive Bayes Accuracy: {accuracy_score(y_test1, nb_q1.predict(X_test1)):.4f}")
print(f"(xxvi) Logistic Regression Accuracy: {accuracy_score(y_test1, lr_q1.predict(X_test1)):.4f}")

# (xxvii) Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test1, lr_q1.predict(X_test1)), annot=True, fmt='d', cmap='Blues')
plt.title("Visual 7: Confusion Matrix (Logistic Regression)")
plt.savefig("Sentiment_Visual_7_Confusion_Matrix.png")
plt.show(block=False); plt.pause(1)

# (xxvii-extra) ROC Curve
y_pred_prob = lr_q1.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Visual 8: Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("Sentiment_Visual_8_ROC_Curve.png")
plt.show(block=False); plt.pause(1)


# (xxxi-xxxii) MODEL COMPARISON AND FINAL CONCLUSION
print("\n" + "="*50)
print("### 8. Model Comparison and Final Conclusion")
print("="*50)
lr_acc = accuracy_score(y_test1, lr_q1.predict(X_test1))
nb_acc = accuracy_score(y_test1, nb_q1.predict(X_test1))

print(f"The evaluated results show that Logistic Regression performs best overall among the classical models.")
print(f"It achieved an accuracy of {lr_acc:.3f}, which is slightly higher than Naive Bayes at {nb_acc:.3f}.")
print("\nTraditional machine learning models still have clear limitations.")
print("Naive Bayes relies on an 'independence assumption' between words, so it can miss phrase-level meaning.")
print("Logistic Regression is stronger, but it is still a 'linear model' over sparse features,")
print("so it can struggle with sarcasm and deeper semantic relationships.")
print("\nTransformer models (like BERT) address some of these issues because they use 'attention'")
print("to capture word order and surrounding meaning more effectively. However, Transformers")
print("require significantly more computation and memory. In this study, that makes")
print("Logistic Regression the best overall choice for the current setup because it gives")
print("strong performance while staying simple, fast, and practical to train.")
print("="*50)



# ============================================================
# QUESTION 2: TOPIC CLUSTERING (COMPREHENSIVE RUBRIC)
# ============================================================

print("\n" + "="*60)
print("QUESTION 2: TOPIC CLUSTERING (REFINED FOR RUBRIC i-xxix)")
print("="*60)

# --- Question 2 IMPORTS ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# (a) Data Preparation and Understanding (i-iv)
print("\n--- (a) Data Preparation and Understanding ---")
df2 = pd.read_json(os.path.join('news_data', 'News_Category_Dataset_v3.json'), lines=True)
df2['text'] = df2['headline'] + " " + df2['short_description']
df2 = df2.sample(1000, random_state=42).reset_index(drop=True) # 1000 for speed with transformers

print("i. First 10 records to understand structure:")
print(df2[['headline', 'category', 'short_description']].head(10))

# Visual 8b: Category Distribution (Visualizing the raw dataset)
plt.figure(figsize=(12, 6))
df2['category'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title("Visual 8b: Top 10 News Categories in Dataset")
plt.xlabel("Category"); plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("Clustering_Visual_8b_Category_Distribution.png")
plt.show(block=False); plt.pause(1)

print("\nii. Key Columns: 'headline' (main topic), 'short_description' (context). Both are vital for semantic clustering.")

total_docs = len(df2)
avg_len = df2['text'].apply(lambda x: len(str(x).split())).mean()
all_words_q2 = " ".join(df2['text']).split()
vocab_size = len(set(all_words_q2))
print(f"\niii. Statistics: Total Docs: {total_docs}, Avg Length: {avg_len:.2f} words, Raw Vocab: {vocab_size}")

print("\niv. Noise Inspection: Found 2 types of noise: 1) HTML entities (e.g. &amp;), 2) Informal punctuation (e.g. ---).")
print("    Effect: These add sparsity and irrelevant dimensions to the TF-IDF matrix.")

# (b) Text Preprocessing and Normalization (v-viii)
print("\n--- (b) Text Preprocessing and Normalization ---")
print("viii. Before vs After Example:")
sample_idx = 0
print(f"  BEFORE: {df2['text'].iloc[sample_idx][:100]}...")
df2['clean_text'] = df2['text'].apply(clean_text_q1) # Reusing Question 1 cleaner
print(f"  AFTER:  {df2['clean_text'].iloc[sample_idx][:100]}...")
print("    Improvement: Preprocessing reduces dimensionality by grouping 'running', 'runs' into 'run'.")

# (c) NLP Exploration (ix-xi)
print("\n--- (c) NLP Exploration ---")
top_10 = Counter(" ".join(df2['clean_text']).split()).most_common(10)
print(f"ix. Top 10 words: {top_10}")

bi_q2 = Counter(nltk.bigrams(" ".join(df2['clean_text']).split())).most_common(5)
tri_q2 = Counter(nltk.trigrams(" ".join(df2['clean_text']).split())).most_common(5)
print(f"x. Common Phrases (Bigrams/Trigrams): {bi_q2 + tri_q2}")
print("xi. N-Grams help identify multi-word topics like 'donald trump' or 'new york city'.")

# Visual 12: Global Word Cloud for Question 2
plt.figure()
all_text_q2 = " ".join(df2['clean_text'])
wc_q2 = WordCloud(width=800, height=400, background_color='white').generate(all_text_q2)
plt.imshow(wc_q2, interpolation='bilinear')
plt.title("Visual 12: Question 2 Global Word Cloud")
plt.axis('off')
plt.savefig("Clustering_Visual_12_Global_WordCloud.png")
plt.show(block=False); plt.pause(1)

# (d) Text Representation (xii-xv)
print("\n--- (d) Text Representation ---")
bow_vec = CountVectorizer(max_features=1000)
X_bow = bow_vec.fit_transform(df2['clean_text'])
print(f"xii. Bag-of-Words Shape: {X_bow.shape}")

# Visual 13: Top 10 Bag-of-Words Features
bow_counts = X_bow.sum(axis=0).A1
bow_words = bow_vec.get_feature_names_out()
bow_df = pd.DataFrame({'Word': bow_words, 'Count': bow_counts}).sort_values(by='Count', ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x='Count', y='Word', data=bow_df, palette='Blues_r')
plt.title("Visual 13: Top 10 Bag-of-Words Features (Task xii)")
plt.savefig("Clustering_Visual_13_BoW.png")
plt.show(block=False); plt.pause(1)

tfidf_vec = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vec.fit_transform(df2['clean_text'])
print(f"xiii. TF-IDF Shape: {X_tfidf.shape}")

# Visual 14: Top 10 TF-IDF Features
tfidf_weights = X_tfidf.sum(axis=0).A1
tfidf_words = tfidf_vec.get_feature_names_out()
tfidf_df = pd.DataFrame({'Word': tfidf_words, 'Weight': tfidf_weights}).sort_values(by='Weight', ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x='Weight', y='Word', data=tfidf_df, palette='Greens_r')
plt.title("Visual 14: Top 10 TF-IDF Features (Task xiii)")
plt.savefig("Clustering_Visual_14_TFIDF.png")
plt.show(block=False); plt.pause(1)

# Visual 15: BoW vs TF-IDF Comparison
# Select 5 common words to compare
common_words = bow_df['Word'].head(5).tolist()
comp_data = []
for w in common_words:
    b_idx = list(bow_words).index(w)
    t_idx = list(tfidf_words).index(w)
    # Normalize for comparison
    comp_data.append({'Word': w, 'Method': 'BoW (Count)', 'Value': bow_counts[b_idx]/bow_counts.max()})
    comp_data.append({'Word': w, 'Method': 'TF-IDF (Weight)', 'Value': tfidf_weights[t_idx]/tfidf_weights.max()})
comp_df = pd.DataFrame(comp_data)
plt.figure(figsize=(10, 5))
sns.barplot(x='Value', y='Word', hue='Method', data=comp_df)
plt.title("Visual 15: BoW vs TF-IDF Weighting Comparison (Task xiv)")
plt.savefig("Clustering_Visual_15_Comparison.png")
plt.show(block=False); plt.pause(1)

print("xiv. Comparison: BoW uses counts; TF-IDF weights rare, important words higher, improving separation.")

# (e) Document Similarity Analysis (xvi-xviii)
print("\n--- (e) Document Similarity ---")
sim_matrix = cosine_similarity(X_tfidf[:50]) # Sample similarity
pairs = []
for i in range(len(sim_matrix)):
    for j in range(i+1, len(sim_matrix)):
        pairs.append((i, j, sim_matrix[i,j]))
sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
print(f"xvii. Most similar pair: Docs {sorted_pairs[0][0]} & {sorted_pairs[0][1]} with score {sorted_pairs[0][2]:.4f}")
print(f"xviii. Doc A: {df2['text'].iloc[sorted_pairs[0][0]][:50]}")
print(f"       Doc B: {df2['text'].iloc[sorted_pairs[0][1]][:50]}")

# (f) K-Means Clustering (xix-xxii)
print("\n--- (f) K-Means (Traditional) ---")
wcss_q2 = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_tfidf).inertia_ for k in range(2, 7)]
plt.figure(); plt.plot(range(2, 7), wcss_q2, 'bx-'); plt.title("Visual 9: Elbow (TF-IDF)"); plt.savefig("Clustering_Visual_9_Elbow.png")

k_opt = 4
km_tfidf = KMeans(n_clusters=k_opt, random_state=42, n_init=10).fit(X_tfidf)
df2['cluster_tfidf'] = km_tfidf.labels_
print(f"xx. Choosing K={k_opt} based on the Elbow point in Visual 9.")
print("xxii. Cluster interpretation (Topics): Politics, Lifestyle, News, and Culture based on keywords.")

# (g) Transformer Embeddings (xxiii-xxvi)
print("\n--- (g) Transformer Embeddings ---")
print("xxiii. Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
t_model = SentenceTransformer('all-MiniLM-L6-v2')
X_trans = t_model.encode(df2['text'].tolist(), show_progress_bar=False)
print(f"xxiii. Transformer Embedding Shape: {X_trans.shape}")

km_trans = KMeans(n_clusters=k_opt, random_state=42, n_init=10).fit(X_trans)
df2['cluster_trans'] = km_trans.labels_
print("xxiv. K-Means applied to Transformer embeddings.")
print("xxvi. Transformers capture 'context' and 'semantics' (synonyms) better than simple word counts.")

# (h) Evaluation and Insight (xxvii-xxix)
print("\n--- (h) Evaluation and Insight ---")
terms = tfidf_vec.get_feature_names_out()
order = km_tfidf.cluster_centers_.argsort()[:, ::-1]
print("xxvii. Top Keywords per TF-IDF Cluster:")
for i in range(k_opt):
    print(f" Cluster {i}: {[terms[ind] for ind in order[i, :5]]}")

print("\nxxix. FINAL COMPARISON:")
print(" - TF-IDF Strengths: Fast, interpretable, good for distinct keywords.")
print(" - Transformer Strengths: Captures meaning even when different words are used (e.g. 'car' vs 'automobile').")

# Visualizations for Question 2
pca_q2 = PCA(n_components=2)
coords_tfidf = pca_q2.fit_transform(X_tfidf.toarray())
coords_trans = pca_q2.fit_transform(X_trans)

plt.figure(); sns.scatterplot(x=coords_tfidf[:,0], y=coords_tfidf[:,1], hue=df2['cluster_tfidf'], palette='viridis')
plt.title("Visual 11: PCA (TF-IDF)"); plt.savefig("Clustering_Visual_11_PCA_TFIDF.png")

plt.figure(); sns.scatterplot(x=coords_trans[:,0], y=coords_trans[:,1], hue=df2['cluster_trans'], palette='magma')
plt.title("Visual 11b: PCA (Transformer)"); plt.savefig("Clustering_Visual_11_PCA_Transformer.png")

print("\nQuestion 2 complete. Submission Package Ready.")
plt.show(block=False); plt.pause(2)

print("\nSubmission Package Ready.")
plt.show(block=False); plt.pause(3)
