# Disaster Response Pipeline Project

### Project Goals

- Analyze disaster data from [Appen](appen.com) (formally Figure 8) to build a model for an API that classifies disaster messages.

 - Create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

- Build a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://localhost:3000/

### Screenshot
<img width="945" alt="image" src="https://github.com/senorpinatta/DisasterReponseTimeline/assets/8880195/24363bb0-734f-4d33-b626-ffebc2709366">

### Acknowledgments
* [Appen](appen.com) (formally Figure-Eight) for providing the dataset.
