import gdown


if __name__ == "__main__":
    url = 'https://drive.google.com/uc?id=' + "1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar"
    output = 'HomologyTAPE.zip'
    gdown.download(url, output)
