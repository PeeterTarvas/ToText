import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile


eminst_class_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
               12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
               23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
               34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
               45: 'r', 46: 't'}


def change_to_project_root_dir() -> str:
    current_directory_path: list[str] = os.getcwd().split('\\')
    project_root_dir_as_list: list[str] = current_directory_path[:current_directory_path.index("src")]
    project_root_dir = "\\".join(project_root_dir_as_list)
    os.chdir(project_root_dir)
    return project_root_dir


def extract_zip(zip_path: str, to_path: str):
    # Extract the ZIP file
    print(os.getcwd())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(to_path)


def calculate_row_estimate(csv_path, size_in_mb) -> int:
    sample_size = 1000
    sample_df = pd.read_csv(csv_path, nrows=sample_size)
    avg_row_size = sample_df.memory_usage(deep=True).sum() / sample_size
    target_size_mb = size_in_mb
    target_size_bytes = target_size_mb * 1024 * 1024
    estimated_rows = int(target_size_bytes / avg_row_size)
    return estimated_rows


def load_emnist(folder_path: str, partial: bool = False, size_in_mb: int = 300):
    if partial:
        rows = calculate_row_estimate(folder_path + '/emnist-byclass-train.csv', size_in_mb)
        emnist_train_df: pd.DataFrame = pd.read_csv(folder_path + '/emnist-byclass-train.csv', delimiter=',',
                                                    header=None, nrows=rows)
    else:
        emnist_train_df: pd.DataFrame = pd.read_csv(folder_path + '/emnist-byclass-train.csv', delimiter=',', header=None)
    return emnist_train_df

def filter_by_labels(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    print(df[0])
    return df[df[0].isin(labels.keys())].reset_index(drop=True)

def format_to_train(df: pd.DataFrame, mapping: dict[int: str]):
    x_train = df.drop(columns=[0], axis=1)
    y_train = df[0].map(mapping)
    return x_train, y_train


def show_images(x, y, num_images):
    num_cols = num_images
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 2, num_rows * 2))

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x.iloc[i].values.reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Display a sample image
def show_sample(images, labels, index):
    plt.imshow(images[index], cmap='gray')
    plt.title(f'Label: {labels[index][0]}')
    plt.show()


if __name__ == '__main__':
    change_to_project_root_dir()
    # extract_zip("./resources/tars/eminst.zip", "resources/data/")
    df = load_emnist('resources/data', True, 10)
    df = filter_by_labels(df, eminst_class_mapping)
    print(df.head())
    x_train, y_train = format_to_train(df, eminst_class_mapping)
    show_images(x_train, y_train, 10)
