import pandas as pd
import requests
from scipy.io import arff
from io import StringIO


def chunker(seq, size):  # from https://stackoverflow.com/a/434328
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def fetch_arff_file_to_dataframe(url_to_arff_file):
    server_response = requests.get(url_to_arff_file)

    # Check if call correct
    if server_response.status_code != 200:
        # FIXME implement bad response handling...
        raise ConnectionError("Unknown status code {} for requesting of arff file {}.".format(
            server_response.status_code, url_to_arff_file))

    # Parse Arff and format to dataframe
    dataset, meta_data = arff.loadarff(StringIO(server_response.text))
    df_dataset = pd.DataFrame(dataset)

    return df_dataset, meta_data
