import streamlit as st 
st.title("WELCOME TO PREDICTION TASK")
   
    
menu=['CLUSTERING','CLASSIFICATION','TASK 3']
submenu=['plot','prediction']
choice=st.sidebar.selectbox('SELECT ANY ONE TASK',['TASK1','TASK2','TASK3'])
if choice =='TASK1':
        st.subheader(":red[TASK1(ClUSTERING MODEL)]")
        st.write(":green[Provide below feature value]")
        first,last=st.columns(2)
        with first:
            st.number_input("T1")
        with last:
            st.number_input("T2")
        first,last=st.columns(2)
        with first:
            st.number_input("T3")
        with last:
            st.number_input("T4")
        first,last=st.columns(2)
        with first:
            st.number_input("T5")
        with last:
            st.number_input("T6")
        first,last=st.columns(2)
        with first:
            st.number_input("T7")
        with last:
            st.number_input("T8")
        first,last=st.columns(2)
        with first:
            st.number_input("T9")
        with last:
            st.number_input("T10")
        first,last=st.columns(2)
        with first:
            st.number_input("T11")
        with last:
            st.number_input("T12")
        first,last=st.columns(2)
        with first:
            st.number_input("T13")
        with last:
            st.number_input("T14")
        first,last=st.columns(2)
        with first:
            st.number_input("T15")
        with last:
            st.number_input("T16")
        first,last=st.columns(2)
        with first:
            st.number_input("T17")
        with last:
            st.number_input("T18")
        first,last=st.columns(2)
        
        #CLUSTERING PROGRAM
        
        
        
        # Import necessary libraries
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
       
        # Load the datasets
        train_data = pd.read_excel('train.xlsx')
        test_data = pd.read_excel('test.xlsx')
 
        # Display the first few rows and shape of the datasets
        print("Training Data Head:")
        print(train_data.head())
        print("Training Data Shape:", train_data.shape)

        print("\nTest Data Head:")
        print(test_data.head())
        print("Test Data Shape:", test_data.shape)
        
        # Preprocess the data
        # Handle missing values (if any)
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        # Select only numeric columns for clustering
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        train_data_numeric = train_data[numeric_cols]
        test_data_numeric = test_data[numeric_cols]

        # Display the first few rows and shape of the numeric datasets
        print("\nNumeric Training Data Head:")
        print(train_data_numeric.head())
        print("Numeric Training Data Shape:", train_data_numeric.shape)

        print("\nNumeric Test Data Head:")
        print(test_data_numeric.head())
        print("Numeric Test Data Shape:", test_data_numeric.shape)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_train_data = scaler.fit_transform(train_data_numeric)
        scaled_test_data = scaler.transform(test_data_numeric)
 
        # Determine the optimal number of clusters using the Elbow method and Silhouette score
        inertia_values = []
        silhouette_scores = []
        davies_bouldin_scores = []
        k_values = range(2, 11)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_train_data)
    
            inertia_values.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_train_data, kmeans.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(scaled_train_data, kmeans.labels_))

 
        # Plotting the Elbow Method
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, inertia_values, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        
        
        # Plotting the Silhouette Scores
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores For Different k')
        plt.show()

        # Plotting the Davies-Bouldin Scores
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, davies_bouldin_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index For Different k')
        plt.show()

        # Choose an optimal number of clusters (from the plots, choose the best k)
        optimal_k = 3  # This should be chosen based on the Elbow and Silhouette analysis

        # Fit the model with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(scaled_train_data)

        # Add the cluster labels to the train dataset
        train_data['Cluster'] = kmeans.labels_

        # Recalculate performance metrics for the optimal number of clusters
        silhouette_avg = silhouette_score(scaled_train_data, kmeans.labels_)
        inertia = kmeans.inertia_
        davies_bouldin = davies_bouldin_score(scaled_train_data, kmeans.labels_)

        print("\nPerformance Metrics for Training Data with Optimal k:")
        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Inertia: {inertia}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")

        # Function to predict the cluster for a new data point
        def predict_cluster(new_data_point, scaler, kmeans):
           # Scale the new data point
           scaled_new_data_point = scaler.transform([new_data_point])
           # Predict the cluster
           cluster = kmeans.predict(scaled_new_data_point)
           return cluster[0]

        # Predict clusters for the test dataset
        test_data['Cluster'] = test_data_numeric.apply(lambda row: predict_cluster(row.values, scaler, kmeans), axis=1)


        # Function to explain why a data point belongs to a particular cluster
        def explain_cluster(new_data_point, cluster, kmeans, scaler):
            cluster_centers = kmeans.cluster_centers_
            scaled_new_data_point = scaler.transform([new_data_point])
            distances = np.linalg.norm(scaled_new_data_point - cluster_centers, axis=1)
    
            explanation = f"The new data point is closest to cluster {cluster} with a distance of {distances[cluster]}.\n"
            explanation += f"Cluster {cluster} center: {cluster_centers[cluster]}"
    
            return explanation


        # Example of explaining why a test data point belongs to the predicted cluster
        example_test_point = test_data_numeric.iloc[0, :].values  # Example: Using the first row of the test dataset as an example
        predicted_cluster = test_data.iloc[0, -1]
        explanation = explain_cluster(example_test_point, predicted_cluster, kmeans, scaler)
        print(f"\nThe test data point belongs to cluster: {predicted_cluster}")
        print(explanation)

        # Save the results to an Excel file
        test_data.to_excel('test_dataset_with_clusters.xlsx', index=False)
        # Visualization: Reduce dimensions to 2D using PCA and plot the clusters
        from sklearn.decomposition import PCA
            
        pca = PCA(n_components=2)
        pca_train_data = pca.fit_transform(scaled_train_data)
            
        # Plotting the clusters
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_train_data[:,0], y=pca_train_data[:,1], hue=train_data['Cluster'], palette='viridis', s=100, alpha=0.7)
        plt.title('Clusters Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
        # Display the head and shape of the test data with clusters
        print("\nTest Data with Clusters Head:")
        print(test_data.head())
        print("Test Data with Clusters Shape:", test_data.shape)
        
            
elif choice =='TASK2':
        st.subheader(":red[TASK2(CLASSIFICATION MODEL)]")
        st.write(":green[Provide below feature value]")
        first,last=st.columns(2)
        with first:
            st.number_input("T1")
        with last:
            st.number_input("T2")
        first,last=st.columns(2)
        with first:
            st.number_input("T3")
        with last:
            st.number_input("T4")
        first,last=st.columns(2)
        with first:
            st.number_input("T5")
        with last:
            st.number_input("T6")
        first,last=st.columns(2)
        with first:
            st.number_input("T7")
        with last:
            st.number_input("T8")
        first,last=st.columns(2)
        with first:
            st.number_input("T9")
        with last:
            st.number_input("T10")
        first,last=st.columns(2)
        with first:
            st.number_input("T11")
        with last:
            st.number_input("T12")
        first,last=st.columns(2)
        with first:
            st.number_input("T13")
        with last:
            st.number_input("T14")
        first,last=st.columns(2)
        with first:
            st.number_input("T15")
        with last:
            st.number_input("T16")
        first,last=st.columns(2)
        with first:
            st.number_input("T17")
        with last:
            st.number_input("T18")
        first,last=st.columns(2)
        
        #classification program
        
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Load train and test datasets
        train_data = pd.read_excel('train.xlsx')
        test_data = pd.read_excel('test.xlsx')
        
        # Identify features and target variable in train dataset
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2, random_state=42)

        # Initialize classifiers
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        lr_classifier = LogisticRegression(max_iter=1000)
        svc_classifier = SVC()

        # Train classifiers
        rf_classifier.fit(X_train, y_train)
        
        
        from sklearn.preprocessing import StandardScaler

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr_classifier.fit(X_train_scaled, y_train)
        svc_classifier.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Evaluate classifiers
        classifiers = {'Random Forest': rf_classifier, 'Logistic Regression': lr_classifier, 'Support Vector Classifier': svc_classifier}
        results = {}

        for name, classifier in classifiers.items():
            try:
                if name == 'Random Forest':
                    classifier.fit(X_train, y_train)  # Ensure RandomForestClassifier is fitted
                    y_pred = classifier.predict(X_test_scaled)
                elif name == 'Logistic Regression':
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test_scaled)  # Use scaled data for LR
                else:
                    classifier.fit(X_train, y_train)  # Fit SVC classifier
                    y_pred = classifier.predict(X_test)
        
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')  # Choose an appropriate average parameter
                recall = recall_score(y_test, y_pred, average='weighted')  # Choose an appropriate average parameter
                f1 = f1_score(y_test, y_pred, average='weighted')  # Choose an appropriate average parameter
                results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
            except Exception as e:
                print(f"An error occurred while evaluating classifier {name}: {e}")

        
            # Compare results
            results_df = pd.DataFrame(results).transpose()
            print(results_df)
        
              
            # Select the best classifier based on your criteria (e.g., highest accuracy)
            best_classifier_name = results_df['Accuracy'].idxmax()
            best_classifier = classifiers[best_classifier_name]  
        
            # Train the best classifier on the entire dataset
            best_classifier.fit(X_train, y_train)
            
            # Make predictions using the trained SVC classifier
            bt2=st.button("Predict")
            if bt2:
                predictions = best_classifier.predict(X_test)
                st.write("Classifier predict it into class",predictions)
            
            
            # Share target values for the test data
            # Assuming X_test is a DataFrame and y_pred is the predicted target values
            # Share target values for the test data
            test_target_values = pd.DataFrame({'Predicted_Target': y_pred})
            test_target_values.to_excel('predicted_target_values.xlsx', index=False)
            
            # Calculate and print train accuracy
            train_accuracy = accuracy_score(y_train, best_classifier.predict(X_train))
            print("Train Accuracy:", train_accuracy)
   
         
elif choice =='TASK3':
        st.subheader(":red[TASK3(TIME-SERIES)]")
        

        import pandas as pd

        # Assuming your dataframe is named df
        df=pd.read_excel('rawdata.xlsx')

        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Task 1: Datewise total duration for each inside and outside
        inside_duration = df[df['position'].str.lower() == 'inside'].groupby('date')['number'].sum().reset_index(name='Inside Duration')
        outside_duration = df[df['position'].str.lower() == 'outside'].groupby('date')['number'].sum().reset_index(name='Outside Duration')

        # Task 2: Datewise number of picking and placing activity done
        picking_count = df[df['activity'].str.lower() == 'picked'].groupby('date').size().reset_index(name='Picking Count')
        placing_count = df[df['activity'].str.lower() == 'placed'].groupby('date').size().reset_index(name='Placing Count')

        # Merge results
        result = pd.merge(inside_duration, outside_duration, on='date', how='outer')
        result = pd.merge(result, picking_count, on='date', how='outer')
        result = pd.merge(result, placing_count, on='date', how='outer')

        # Fill NaN values with 0
        result.fillna(0, inplace=True)

        # Convert duration columns to integers
        result['Inside Duration'] = result['Inside Duration'].astype(int)
        result['Outside Duration'] = result['Outside Duration'].astype(int)
        bt3=st.button("Predict")
        if bt3:  
          st.write("Result is ",result)
        # Save the result to an Excel file with 'date' column included
        result.to_excel('output.xlsx', index=False, engine='xlsxwriter')

        print("Output saved to output.xlsx")

        
else:
    pass
    
    

