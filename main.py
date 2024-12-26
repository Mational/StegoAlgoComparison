from StegoCrypt import StegoCrypt

if __name__ == '__main__':
    plaintext = (
        "\"Być czy nie być, oto jest pytanie!\" Powiedział Hamlet w sztuce Szekspira."
    )

    # Initialize the StegoCrypt class with alpha and threshold
    steg = StegoCrypt(alpha=2.0, threshold=0.5)

    cover_image_path = "Lena_Image.png"
    stego_image_path = "Lena_Image_Stego.png"

    # Hide data in the cover image
    logistic_map_seeds, mersenne_keys_seeds = steg.hide_data_in_stego_image(
        plaintext,
        cover_image_path,
        stego_image_path
    )

    # Extract the hidden data from the stego image
    decoded_message = steg.extract_message_from_stego_image(
        stego_image_path,
        mersenne_keys_seeds,
        logistic_map_seeds
    )

    print("Decrypted message:")
    print(decoded_message)