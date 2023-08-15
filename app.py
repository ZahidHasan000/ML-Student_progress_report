# from hasan import plot
# from flask import Flask, jsonify
# import pandas as pd
# from pymongo import MongoClient

# client = MongoClient("mongodb://localhost:27017/")
# csv_file = 'StudentProgress.csv'
# df = pd.read_csv(csv_file)

# # Convert DataFrame to JSON format
# data_json = df.to_dict(orient='records')

# db = client["MachineLearning"]
# db.StudentProgress.insert_many(data_json)

# mycollection = db["StudentProgress"]

# one = mycollection.find_one()
# # print(one)

# all = mycollection.find()
# # print(all)

# # for row in all:
# #     # print(row)

# #     list_cursor = list(all)
# #     # print(list_cursor)

# #     df = pd.DataFrame(list_cursor)

# #     # print(df.head)
# #     # print(df.corr)
# #     df.head()


# client = MongoClient("mongodb://localhost:27017/")
# db = client["MachineLearning"]
# collection = db["StudentProgress"]

# app = Flask(__name__)

# # API endpoint to fetch all data


# @app.route('/api/students', methods=['GET'])
# def get_students():
#     cursor = collection.find({}, {'_id': 0})
#     data = list(cursor)
#     df = pd.DataFrame(data)

#     analysis_result = plot(df)
#     # students = list(cursor)
#     return jsonify(analysis_result)

# # API endpoint to fetch a specific student by their ID


# @app.route('/api/students/<int:student_id>', methods=['GET'])
# def get_student(student_id):
#     student = collection.find_one({'student_id': student_id}, {'_id': 0})
#     if student:
#         return jsonify(student)
#     else:
#         return jsonify({'message': 'Student not found'}), 404


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, jsonify
# import pandas as pd
# from pymongo import MongoClient
# from hasan import plot  # Import the necessary functions from hasan.py

# client = MongoClient("mongodb://localhost:27017/")
# db = client["MachineLearning"]
# collection = db["StudentProgress"]

# app = Flask(__name__)

# # ... Your MongoDB code for inserting data ...

# # API endpoint to perform data analysis and return the results


# @app.route('/api/analysis', methods=['GET'])
# def perform_analysis():
#     # Get the data from the MongoDB collection
#     cursor = collection.find({}, {'_id': 0})
#     data = list(cursor)
#     df = pd.DataFrame(data)

#     # print(df.head)

#     # Call the data analysis function from hasan.py
#     analysis_result = plot(df)

#     # Return the analysis result as JSON
#     return jsonify(analysis_result)


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, jsonify
# import pandas as pd
# from pymongo import MongoClient
# from hasan import plot  # Import the necessary functions from hasan.py
# import os

# client = MongoClient("mongodb://localhost:27017/")
# db = client["MachineLearning"]
# collection = db["StudentProgress"]

# app = Flask(__name__)

# # ... Your MongoDB code for inserting data ...

# # API endpoint to perform data analysis and return the results


# @app.route('/api/analysis', methods=['GET'])
# def perform_analysis():
#     # Get the data from the MongoDB collection
#     cursor = collection.find({}, {'_id': 0})
#     data = list(cursor)
#     df = pd.DataFrame(data)

#     # Ensure the 'static' directory exists
#     if not os.path.exists('static'):
#         os.makedirs('static')

#     # Call the data analysis function from hasan.py
#     plot_path = plot(df)

#     # Return the file path of the saved figure and the analysis result as JSON
#     response = {
#         'plot_path': plot_path
#     }

#     return jsonify(response)


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, jsonify
import pandas as pd
from pymongo import MongoClient
from hasan import plot  # Import the necessary functions from hasan.py
import os
# import threading


client = MongoClient("mongodb://localhost:27017/")
db = client["MachineLearning"]
collection = db["StudentProgress"]

app = Flask(__name__)

# ... Your MongoDB code for inserting data ...

# API endpoint to perform data analysis and return the results


@app.route('/api/analysis', methods=['GET'])
def perform_analysis():
    # Get the data from the MongoDB collection
    cursor = collection.find({}, {'_id': 0})
    data = list(cursor)
    df = pd.DataFrame(data)

    # Ensure the 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Call the data analysis function from hasan.py
    plot_path = plot(df)

    # Return the file path of the saved figure and the analysis result as JSON
    response = {
        'plot_path': plot_path
    }
    return jsonify(response)

    # # Return a response to the client immediately
    # response = {
    #     # 'plot_path': plot_paths
    #     'message': 'Analysis started. Plots will be generated in the background.'
    # }

    # return jsonify(response)

# API endpoint to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(threaded=True)
