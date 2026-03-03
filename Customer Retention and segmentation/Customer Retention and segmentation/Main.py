import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("A Hybrid Unsupervised Learning Approach for Segmenting High and Low Revenue Customers in E-commerce") 
main.geometry("1300x1200")

global filename
global x_train, y_train, x_test, y_test
global X, Y
global le
global dataset
global classifier, cnn_model, rfc
accuracy = []
precision = []
recall = []
fscore = []
silhouette_scores = []

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n')
    dataset = pd.read_csv(filename)
    dataset = dataset.drop('custid', axis=1)
    text.insert(END, str(dataset) + "\n\n")

def preprocessDataset():
    global X, Y, y
    global le
    global dataset
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)
    #text.insert(END, str(dataset.head()) + "\n\n")


    dataset['created'] = pd.to_datetime(dataset['created'], format='%d-%m-%Y', errors='coerce')
    dataset['firstorder'] = pd.to_datetime(dataset['firstorder'], format='%d-%m-%Y %H:%M', errors='coerce')
    dataset['lastorder'] = pd.to_datetime(dataset['lastorder'], format='%d-%m-%Y %H:%M', errors='coerce')
    dataset['created_year'] = dataset['created'].dt.year
    dataset['created_month'] = dataset['created'].dt.month
    dataset['created_day'] = dataset['created'].dt.day
    dataset = dataset.drop(['created'], axis=1)
    
    dataset['firstorder_year'] = dataset['firstorder'].dt.year
    dataset['firstorder_month'] = dataset['firstorder'].dt.month
    dataset['firstorder_day'] = dataset['firstorder'].dt.day
    dataset = dataset.drop(['firstorder'], axis=1)
    
    dataset['lastorder_year'] = dataset['lastorder'].dt.year
    dataset['lastorder_month'] = dataset['lastorder'].dt.month
    dataset['lastorder_day'] = dataset['lastorder'].dt.day
    dataset = dataset.drop(['lastorder'], axis=1)

    #text.insert(END, str(dataset.head()) + "\n\n")
    text.insert(END, str(dataset) + "\n\n")

    non_numeric_columns = dataset.select_dtypes(exclude = ['int', 'float']).columns
    for col in non_numeric_columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])


    #dataset['favday'] = le.fit_transform(dataset['favday'])
    #dataset['city'] = le.fit_transform(dataset['city'])
    text.insert(END, str(dataset.head()) + "\n\n")
    dataset.dropna(inplace=True)
    X = dataset.drop('retained', axis=1)
    
    Y = dataset['retained']
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, y = smote.fit_resample(X, Y)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

    #sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='retained', data=dataset, palette="Set3")
    plt.title("Count Plot")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()
    

def analysis():
    global y
    #sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x = y, palette="Set3")
    plt.title("Count Plot After SMOTE")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

def pcaVisualization():
    global dataset
    text.delete('1.0', END)
    
    dataset['revenue'] = dataset['avgorder']
    non_zero_revenue = dataset[dataset['revenue'] > 0]['revenue']
    
    if len(non_zero_revenue) == 0:
        text.insert(END, "Error: No customers with non-zero revenue found.\n")
        return
    
    revenue_threshold = non_zero_revenue.quantile(0.5)
    dataset['revenue_category'] = np.where(dataset['revenue'] >= revenue_threshold, 'High Revenue', 'Low Revenue')
    
    X_pca = dataset.drop(['retained', 'revenue', 'revenue_category'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    X_pca_transformed = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca_transformed, columns=['PC1', 'PC2'])
    pca_df['Revenue Category'] = dataset['revenue_category'].values
    pca_df['Retained'] = dataset['retained'].values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Revenue Category', style='Retained', 
                    palette={'High Revenue': 'red', 'Low Revenue': 'blue'}, 
                    size='Retained', sizes=(50, 100))
    plt.title('PCA Visualization of High vs Low Revenue Customers')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    text.insert(END, "PCA Visualization completed\n")
    text.insert(END, f"Explained variance ratio: PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}\n")
    text.insert(END, f"Revenue Threshold (50th percentile of avgorder): {revenue_threshold:.2f}\n")
    text.insert(END, f"Number of High Revenue Customers: {len(dataset[dataset['revenue_category'] == 'High Revenue'])}\n")
    text.insert(END, f"Number of Low Revenue Customers: {len(dataset[dataset['revenue_category'] == 'Low Revenue'])}\n")

def classifier():
    global x_train, y_train, x_test, y_test
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=100, random_state=0)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(('ETC', a))
    precision.append(('ETC', p))
    recall.append(('ETC', r))
    fscore.append(('ETC', f))
    text.insert(END, "Extra Tree Classifier Precision : " + str(p) + "\n")
    text.insert(END, "Extra Tree Classifier Recall    : " + str(r) + "\n")
    text.insert(END, "Extra Tree Classifier FMeasure  : " + str(f) + "\n")
    text.insert(END, "Extra Tree Classifier Accuracy  : " + str(a) + "\n\n")
    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Extra Tree Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    report = classification_report(y_test, predict)
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)

def RandomForestclassifier():
    global x_train, y_train, x_test, y_test
    global rfc
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(x_train, y_train)
    predict = rfc.predict(x_test)
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(('RFC', a))
    precision.append(('RFC', p))
    recall.append(('RFC', r))
    fscore.append(('RFC', f))
    text.insert(END, "Random Forest Classifier Precision: " + str(p) + "\n")
    text.insert(END, "Random Forest Classifier Recall: " + str(r) + "\n")
    text.insert(END, "Random Forest Classifier FMeasure: " + str(f) + "\n")
    text.insert(END, "Random Forest Classifier Accuracy: " + str(a) + "\n\n")
    cm = confusion_matrix(y_test, predict)
    report = classification_report(y_test, predict)
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def Prediction():
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    test = test.drop('custid', axis=1)
   # test['favday'] = le.transform(test['favday'])
    test['city'] = le.transform(test['city'])

    test['favday'] = le.fit_transform(test['favday'])
    test['city'] = le.fit_transform(test['city'])
    test['created'] = pd.to_datetime(test['created'], format='%d-%m-%Y')
    test['firstorder'] = pd.to_datetime(test['firstorder'], format='%d-%m-%Y %H:%M', errors='coerce')
    test['lastorder'] = pd.to_datetime(test['lastorder'], format='%d-%m-%Y %H:%M', errors='coerce')
    test['created_year'] = test['created'].dt.year
    test['created_month'] = test['created'].dt.month
    test['created_day'] = test['created'].dt.day
    test = test.drop(['created'], axis=1)
    test['firstorder_year'] = test['firstorder'].dt.year
    test['firstorder_month'] = test['firstorder'].dt.month
    test['firstorder_day'] = test['firstorder'].dt.day
    test = test.drop(['firstorder'], axis=1)
    test['lastorder_year'] = test['lastorder'].dt.year
    test['lastorder_month'] = test['lastorder'].dt.month
    test['lastorder_day'] = test['lastorder'].dt.day
    test = test.drop(['lastorder'], axis=1)
    test.dropna(inplace=True)
    
    for i in range(len(test)):
        input_data = test.iloc[i, :].values.reshape(1, -1)
        predict = rfc.predict(input_data)
        text.insert(END, f'Input data for row {i}: {input_data}\n')
        if predict == 0:
            predicted_data = "Low Revenue"
        elif predict == 1:
            predicted_data = "High Revenue"
        text.insert(END, f'Predicted output for row {i}: {predicted_data}\n')

def kmeansClustering():
    global dataset
    text.delete('1.0', END)
    
    X_cluster = dataset.drop(['retained'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(('K-Means', silhouette))
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    pca_df['Retained'] = dataset['retained'].values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', style='Retained', 
                    palette='Set2', size='Retained', sizes=(50, 100))
    plt.title('K-Means Clustering (2 Clusters)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    dataset['cluster'] = cluster_labels
    cluster_summary = dataset.groupby('cluster')['avgorder'].mean()
    text.insert(END, "K-Means Clustering completed\n")
    text.insert(END, f"Silhouette Score: {silhouette:.4f}\n")
    text.insert(END, "Average avgorder per cluster:\n")
    for cluster, avg in cluster_summary.items():
        text.insert(END, f"Cluster {cluster}: {avg:.2f}\n")
    text.insert(END, f"Number of customers in Cluster 0: {len(dataset[dataset['cluster'] == 0])}\n")
    text.insert(END, f"Number of customers in Cluster 1: {len(dataset[dataset['cluster'] == 1])}\n")


def graph():
    text.delete('1.0', END)
    
    # Classification metrics
    if accuracy:
        df_class = pd.DataFrame([
            ['ETC', 'Precision', precision[0][1]],
            ['ETC', 'Recall', recall[0][1]],
            ['ETC', 'F1 Score', fscore[0][1]],
            ['ETC', 'Accuracy', accuracy[0][1]],
            ['RFC', 'Precision', precision[-1][1]],
            ['RFC', 'Recall', recall[-1][1]],
            ['RFC', 'F1 Score', fscore[-1][1]],
            ['RFC', 'Accuracy', accuracy[-1][1]],
        ], columns=['Algorithm', 'Metric', 'Value'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', hue='Algorithm', data=df_class, palette='Set3')
        plt.title('Classifier Performance Comparison')
        plt.ylabel('Score (%)')
        plt.ylim(0, 100)
        for i, row in df_class.iterrows():
            plt.text(i % 4 + (0.2 if row['Algorithm'] == 'RFC' else -0.2), row['Value'] + 1, f'{row["Value"]:.2f}', 
                     ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    
    # Clustering silhouette scores
    if silhouette_scores:
        df_cluster = pd.DataFrame(silhouette_scores, columns=['Algorithm', 'Silhouette Score'])
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Algorithm', y='Silhouette Score', data=df_cluster, palette='Set2')
        plt.title('Clustering Algorithm Comparison (Silhouette Score)')
        plt.ylabel('Silhouette Score')
        plt.ylim(0, 1)
        for i, row in df_cluster.iterrows():
            plt.text(i, row['Silhouette Score'] + 0.01, f'{row["Silhouette Score"]:.4f}', 
                     ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    
    text.insert(END, "Comparison graphs generated.\n")
    if not accuracy and not silhouette_scores:
        text.insert(END, "No results available to compare.\n")

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='A Hybrid Unsupervised Learning Approach for Segmenting High and Low Revenue Customers in E-commerce', justify=LEFT)
title.config(bg='red', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100, y=5)
title.pack()

font1 = ('times', 13, 'bold')

# Row 1
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=100, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=300, y=100)
preprocessButton.config(font=font1)

analysisButton = Button(main, text="Applying SMOTE", command=analysis)
analysisButton.place(x=500, y=100)
analysisButton.config(font=font1)

knnButton = Button(main, text="Extra Tree Classifier", command=classifier)
knnButton.place(x=700, y=100)
knnButton.config(font=font1)

# Row 2
LRButton = Button(main, text="Random Forest Classifier", command=RandomForestclassifier)
LRButton.place(x=100, y=150)
LRButton.config(font=font1)

predictButton = Button(main, text="Prediction on Test Data", command=Prediction)
predictButton.place(x=400, y=150)
predictButton.config(font=font1)

kmeansButton = Button(main, text="K-Means Clustering", command=kmeansClustering)
kmeansButton.place(x=650, y=150)
kmeansButton.config(font=font1)


graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=300, y=200)
graphButton.config(font=font1)

pcaButton = Button(main, text="PCA Visualization", command=pcaVisualization)
pcaButton.place(x=500, y=200)
pcaButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=700, y=200)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=350)
text.config(font=font1)

main.config(bg='lightpink')
main.mainloop()