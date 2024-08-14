import numpy as np
import pickle
DATA_PATH = ".."
def get_node_number(inPath, outputPath):
    examples = pickle.load(open(inPath + "stat", 'rb'))
    examples = examples[0].astype(int)
    np.save(outputPath+"y.npy", examples)

def path_row_col(inPath, fileName, outputPath):
    examples = pickle.load(open(inPath+fileName, 'rb'))
    # A = np.array([examples[:, 0], examples[:, 2]])
    A = np.array([examples[:, 0], examples[:, 2], examples[:, 3]])
    A = A.astype(int)
    np.save(outputPath+'edge_index.npy', A)

# 验证
def read_npy_file(readFile):
    # read file.npy
    output_data = np.load(readFile)
    print(output_data)

if __name__ == '__main__':
    # for data_name in ["ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT"]:
    for data_name in ["ICEWS14"]:
        inpath = f"{DATA_PATH}/data/{data_name}/"
        outputPath = f"{DATA_PATH}/path_data/{data_name}/"
        fileName = "train.pickle"
        get_node_number(inpath, outputPath)
        path_row_col(inpath, fileName, outputPath)
        # valid
        read_npy_file(outputPath + "y.npy")
        # read_npy_file(outputPath + "edge_index.npy")