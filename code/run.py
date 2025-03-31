import sys
import os

li = list(sys.argv)
model_name = ""

if(li[1] == "manual"):
    os.system("python main.py")
elif (li[1] == "ai"):
    model_name = li[2]
    # Define the relative path
    relative_path = "../model"

    # Get the absolute path from the current working directory
    absolute_path = os.path.abspath(relative_path)

    # List all files inside the "model" directory
    if os.path.exists(absolute_path) and os.path.isdir(absolute_path):
        files = os.listdir(absolute_path)

        model_full_name = f"{model_name}.zip"
        if model_full_name in files:
            os.system(f"python run_ai.py {model_name}")
        else:
            print("Error: Model does not Exist!!")
    else:
        print("\nThe 'model' directory does not exist.")
else:
    print("Invalid File name")





