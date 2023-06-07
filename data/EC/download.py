import gdown


if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=' + "1udP6_90WYkwkvL1LwqIAzf9ibegBJ8rI"
    output = 'ProtFunct.zip'
    gdown.download(url, output)
