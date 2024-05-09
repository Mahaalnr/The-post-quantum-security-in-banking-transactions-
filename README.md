# The-post-quantum-security-in-banking-transactions
Credit card fraud detection using post quantum cryptography (Kyber512)
# Post-Quantum Security in Banking Transactions

Post-quantum security in banking transactions employs advanced cryptographic algorithms designed to protect sensitive credit card information from potential threats posed by powerful quantum computers. This system addresses the growing concern of traditional cryptographic methods becoming vulnerable to quantum attacks.

At its core, the system utilizes post-quantum cryptographic algorithms, which are resistant to attacks from both classical and quantum computers. These algorithms are specifically designed to withstand the computational power of quantum computers and offer a high level of security for credit card data.

## Components:

1. **Data Set**: The system captures credit card data from real transactions made by European cardholders in 2023, with sensitive information removed to ensure privacy and compliance with ethical guidelines.

2. **Preprocessing**: Before applying cryptographic techniques, the system performs preprocessing steps to ensure data integrity and validity. This may involve data validation, sanitization, and error checking to ensure the accuracy of the credit card information.

## Used Algorithms:

1. **ML Method**: Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the accuracy of fraud by learning simple decision rules inferred from the data features.

2. **Post-Quantum Cryptographic Algorithm**: The system employs the Kyber algorithm, an IND-CCA2-secure key encapsulation mechanism (KEM). Kyber is resistant to attacks from both quantum and classical computers and provides a high level of encryption and data integrity.

3. **Encryption**: The credit card information is encrypted using the Kyber post-quantum cryptographic algorithm. Kyber transforms the credit card data into an unreadable format, ensuring security and unintelligibility without the appropriate decryption keys.The submission lists three different parameter sets aiming at different security levels. Specifically, Kyber-512 aims at security roughly equivalent to AES-128, Kyber-768 aims at security roughly equivalent to AES-192, and Kyber-1024 aims at security roughly equivalent to AES-256.

4. **Secure Key Management**: The system incorporates secure key management practices to safeguard encryption and decryption keys. Techniques such as key generation, distribution, storage, and rotation are implemented to protect against unauthorized access to the keys.

5. **Credit Card Detection and Validation**: Mechanisms are included to detect and validate credit card information. This involves verifying the credit card number structure, performing checksum calculations, and cross-referencing with known credit card issuers' databases to ensure validity and authenticity.

6. **Decryption**: When credit card information needs to be accessed or processed for legitimate purposes, the system employs the corresponding decryption algorithm and the authorized decryption key to transform the encrypted data back into its original, readable form.

By employing these post-quantum cryptographic techniques, the credit card detection module provides robust security measures to protect sensitive credit card information from potential attacks, even in the presence of powerful quantum computers. It offers a reliable and future-proof solution to ensure the confidentiality, integrity, and authenticity of credit card transactions in an era of evolving cryptographic threats.

## Advantages of Post-Quantum Cryptography:

1. **Resistance to Quantum Attacks**: Post-quantum cryptographic algorithms are designed to withstand attacks from quantum computers, ensuring the continued security of encrypted data.

2. **Futureproofing**: By adopting post-quantum cryptographic techniques, organizations can mitigate the risk of their encrypted data being compromised in the future, ensuring long-term security.

3. **Compatibility with Existing Infrastructure**: Post-quantum cryptographic algorithms are designed to be compatible with existing cryptographic infrastructure, allowing for a smooth transition without significant changes to the overall architecture.

4. **Security Margins**: Post-quantum cryptographic algorithms often provide larger security margins compared to classical counterparts, ensuring sufficient security even as quantum computing technology advances.

However, it's worth noting that post-quantum cryptographic algorithms may have different performance characteristics compared to classical algorithms. Some post-quantum algorithms may be computationally more demanding, impacting efficiency in practical applications. Ongoing research aims to optimize these algorithms for real-world deployment.

In summary, the primary advantage of post-quantum cryptographic techniques lies in their resistance against quantum attacks, ensuring the continued security of encrypted data in a future where quantum computers become more powerful. While efficiency may vary depending on the specific algorithm, these techniques provide a reliable and future-proof solution for protecting sensitive information in the quantum era.
