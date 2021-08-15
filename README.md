# PyTorch Visual Dialog (a.k.a Ocubot)

Code repository for the final year project focusing on improving visual dialog by reducing modality biases.

> Paper titled "Improving Image-based Dialog by Reducing Modality Biases" accompanying this project has been accepted at the [ICACDS 2021](https://icacds.com/)

**Goal:** To develop an AI agent capable of having a conversation with users about images.

This project uses the [VisDial 1.0 dataset](https://visualdialog.org/data)

### Modality Biases in some of the current Visual Dialog Models

- In the experimentation phase, we observed that some Visual Dialog models tend to focus much more on the dialog history.
- Thus,the answers generated for the current question of the user are not very relevant in all the cases.
- In some cases, we observed that some models were more biased rowards prominent image features and thereby failed to answer questions about finer details like minor features in the image.

#### An example to understand Modality Biases

<img src="https://drive.google.com/uc?export=view&id=143X2N4jrjejmeZ3Ot0_oQfp1HS4v72CY" alt="parking_meter_example"/>

### Our Approach to Reduce these Modality Biases

- There have been approaches to reduce modality biases towards dialog history in the generated responses.
- These approaches tend to reduce attention over dialog history.
- Our project is aimed at reducing the dialog history bias without compromising the attention over dialog history.
- Our approach provides more visual context to the model by generating dense object-level captions which describe the subtleties in the image.
- We have used [DenseCap](https://cs.stanford.edu/people/karpathy/densecap/) proposed by Justin Johnson et al. to generate these dense captions.
- By attending to these Dense Captions, our model generates answers that are more relevant to the current question because the model has additional visual context.
- Also, the bias of existing Visual Dialog models towards prominent image-level features is also reduced by this approach as Dense Captions describe each entity in the image, thus improving the attention over Dense Captions enables the model to generate more relevant answers for questions about the subtleties and minor features present in the image.

### Team Members

- [Jay Gala](https://jaygala24.github.io/)
- [Hrishikesh Shenai](https://www.linkedin.com/in/hrishikesh-shenai-24a32518b/)
- [Pranjal Chitale](https://in.linkedin.com/in/pranjalchitale)
- [Kaustubh Kekre](https://kaustubhkekre.github.io/)
