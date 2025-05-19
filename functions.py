def visualize_data_distribution(df, column_name, y = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column_name, y=y)
    if y is not None:
        plt.title(f'Distribution of {column_name} with respect to {y}')
        plt.ylabel(y)
    else:
        plt.title(f'Distribution of {column_name}')
        plt.ylabel('Count')
    plt.xlabel(column_name)
    plt.xticks(rotation=45)
    plt.show()

# find the mean, min, max, median, std of absenteeism in hours for each absence reason
def boxplot_distribution(df, x, y):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f'{y} by {x}')
    plt.xlabel(x)
    plt.ylabel(f"{y}")
#    plt.xticks(rotation=90)
    plt.show()

def visualize_data_over_time(df, date_column, value_column):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=date_column, y=value_column)
    plt.title(f'{value_column} over Time')
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.show()
    
def bracket_data(data, bins):
    import pandas as pd
    return pd.cut(data, bins=bins, labels=[i for i in range(len(bins)-1)])

def extract_plot_FI(X, y, model, metrics=True, regression=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    if regression:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # plot the feature importances
    if model.__class__.__name__ == 'RandomForestClassifier':
        importances = pipeline.named_steps['model'].feature_importances_
    elif model.__class__.__name__ == 'RandomForestRegressor':
        if metrics:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')
            print(f'R^2 Score: {r2}')
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.title(f'Predicted vs Actual Values ({model.__class__.__name__})')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')       
        return pipeline
    else:
        importances = pipeline.named_steps['model'].coef_[0]
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.title(f'Most Important Features for {y.name} ({model.__class__.__name__})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    
    if metrics:
        plot_confusion_matrix(y_test, y_pred)
    return pipeline

def plot_confusion_matrix(y_true, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def map_reason_category(code):
    if code in range(1, 15):  # I–XIV
        return "medical_physical"
    elif code == 5:
        return "medical_mental"
    elif code in [15, 16, 17]:
        return "pregnancy_related"
    elif code in [18, 21]:
        return "medical_general"
    elif code in [19, 20]:
        return "injury_external"
    elif code in [23, 24, 25, 27, 28]:
        return "routine"
    elif code in [22, 26]:
        return "non_medical"
    else:
        return "unknown"

# median duration of absence per type of absence
def duration_stats(df, reason_col):
    """
    Function to calculate the median duration of absence per type of absence
    """
    import pandas as pd
    # median duration of absence per type of absence
    median_duration = df.groupby(reason_col)['Absenteeism time in hours'].median().round(1)
    mean_duration = df.groupby(reason_col)['Absenteeism time in hours'].mean().round(1)
    # fuse together
    duration_stats = pd.DataFrame({'median': median_duration, 'mean': mean_duration})
    duration_stats['Frequency'] = df[reason_col].value_counts().astype(int)
    sd_duration = df.groupby(reason_col)['Absenteeism time in hours'].std().round(1)
    duration_stats['sd'] = sd_duration
    duration_stats = duration_stats.sort_values(by='median', ascending=False)
    duration_stats = duration_stats.reset_index()
    return duration_stats