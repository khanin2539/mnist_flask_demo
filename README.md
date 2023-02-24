
How to setup an environment:
1.	Assumptions:
a.	Anaconda 3 is available on the machine
b.	Internet is connected 
2.	Setup a virtual environment:  conda create --name certu_test
3.	Install pip if not available: conda install pip
4.	Install required dependencies: pip  install -r requirements.txt


How to run the demo:
1.	Run python app.py. You will see localhost 5000 available at the terminal screen
2.	Spawn the new terminal and go to a test folder containing sample images
3.	If wanting to train, run command  “curl -X POST localhost:5000/train -d "{\"arg1\":0.01,\"arg2\":2,\"arg3\":64}" -H "Content-Type:application/json"” such that arg1-3 are model hyperparameters. Arg1 represents learning rate, arg2 for epoch, and arg3 for batch size. You can change these hyperparameters and any valid format will result to 500 internal error on the user terminal screen. 
4.	If wanting to inference, run command  “curl -X POST localhost:5000/infer -F file=@187.png” such that 187.png represents a sample image. Successful execution will print {"predicted_class":0,"status":200} for example, otherwise 500 internal error to the user. 
5.	For each execution, the server will print the logs from the sqlite3 database such as [('2023-02-24 11:23:55', 'infer', 200), ('2023-02-24 11:23:55', 'infer', 200), ('2023-02-24 11:23:55', 'infer', 500), ('2023-02-24 11:23:55', 'infer', 200), ('2023-02-24 11:23:55', 'train', 200), ('2023-02-24 11:23:55', 'infer', 200), ('2023-02-24 13:26:52', 'infer', 200), ('2023-02-24 13:30:09', 'train', 200)] representing datetime, endpoint, and status. 
