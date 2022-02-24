import bcrypt


def hash(plaintext):
    encoded = bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt())
    return encoded.decode()


if __name__ == "__main__":
    import sys

    print(hash(sys.argv[1]))
