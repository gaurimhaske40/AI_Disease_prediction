
import sys
import os

script_name = sys.argv[0]

argument = sys.argv[1:]

backed_dir = "ai_disease_prediction_backend"


# print(f"Name : {script_name}\nArguments : {argument}")

def get_script_directory():
    """
    Returns the absolute path of the folder where the current script is located.
    """
    return os.path.dirname(os.path.abspath(__file__))

script_dir = get_script_directory()


def InstallingTrainingThenRun():
    try:
        os.chdir(script_dir)
        print("\nInstalling Requirements...\n")
        os.system("pip install -r requirements.txt")
        os.system("cls")
        os.chdir(backed_dir)
        print(os.getcwd())
        print("\nDownloading Datasets Please Wait ...\n")
        os.system("python downloadDatasets.py")
        os.system("cls")
        print("\n\nTraining Models...\n\n")
        os.chdir("training")
        for file in os.listdir():
            if(os.path.isfile(file)):
                os.system(f"python {file}")
            pass
        os.system("cls")
        print("All Done\n\nStarting App")
        RunApp(script_dir=script_dir)
        pass
    except Exception as e:
        pass
    pass

def RunApp(script_dir):
    os.chdir(script_dir)
    os.system("start python runBackend.py")
    os.system("start python runFrontend.py")
    # print(get_script_directory())
    pass


if __name__ == "__main__":
    if len(argument) > 0 :
        if argument[0] == "-i":
            InstallingTrainingThenRun()
        else:
            print("\nInvalid argument provided.\n\n" "Usage:\n" "  -i  To Install Requirements, Train Models, Run The App.\n\n")
    else:
        RunApp(script_dir=script_dir)



