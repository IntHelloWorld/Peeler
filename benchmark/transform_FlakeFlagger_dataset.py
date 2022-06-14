import os
from tqdm import tqdm
import re

"""Transform the raw dataset to specified format as the input of feature extractor."""
if __name__ == "__main__":
    curdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(curdir)
    test_results_dir = "FlakeFlagger/test_results.csv"
    proj_info_dir = "FlakeFlagger/Project_Info.csv"
    proj_dir = "/disk2/FlakyDetect_qyh/projects"
    output = open("FlakeFlagger_benchmark.csv", "w")
    title = ",".join(("Project", "SHA", "Test File Path", "Method Names", "Label"))
    output.write(title + "\n")

    proj_dict = {}
    with open(proj_info_dir, "r") as proj_info:
        for line in proj_info.readlines()[1:]:
            URL, SHA = line.strip("\n").split(",")
            proj_name = URL.split("/")[-1]
            proj_dict[proj_name] = (URL, SHA)
            # clone projects
            os.chdir(proj_dir)
            if not os.path.exists(os.path.join(proj_dir, proj_name)):
                try:
                    os.system("git clone {}".format(URL))
                except Exception as e:
                    print(e)
                    print("clone {} fail!".format(URL))
            # checkout projects
            os.chdir(os.path.join(proj_dir, proj_name))
            os.system("git checkout {}".format(SHA))

    os.chdir(curdir)
    with open(test_results_dir, "r") as csv:
        last_class_name = ""
        last_line = ""
        last_isFlaky = ""
        method_names = []
        for line in tqdm(csv.readlines()[1:], desc="Processing : "):
            proj_name, test, isFlaky, _, _, _, _, _ = line.split(",")
            test_name, method_name = test.split("#")
            if proj_name not in proj_dict.keys():
                continue
            class_name = test_name.split(".")[-1]
            package_name = "/".join(test_name.split(".")[:-2])
            URL, SHA = proj_dict[proj_name]

            cmd = "find {} -name {}".format(os.path.join(proj_dir, proj_name), class_name + ".java")
            with os.popen(cmd) as out:
                java_filepaths = out.readlines()
                for j in java_filepaths:
                    if package_name in j:
                        java_filepath = re.search(r"({}/)(.+)".format(proj_name), j.strip("\n")).group(2)

            if class_name == last_class_name and isFlaky == last_isFlaky:
                method_names.append(method_name)
            else:
                output.write(last_line)
                method_names = [method_name]

            new_line = ",".join((proj_name, SHA, java_filepath, ".".join(method_names), isFlaky + "\n"))
            last_class_name = class_name
            last_isFlaky = isFlaky
            last_line = new_line
        # Append the last line
        new_line = ",".join((proj_name, SHA, java_filepath, ".".join(method_names), isFlaky + "\n"))
        output.write(new_line)
    output.close()
