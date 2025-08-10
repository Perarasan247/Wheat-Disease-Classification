document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const previewImg = document.getElementById("previewImg");
    const uploadBtn = document.getElementById("uploadBtn");
    const clearBtn = document.getElementById("clearBtn");
    const loadingText = document.getElementById("loadingText");
    const resultDiv = document.getElementById("result");

    // Preview selected image
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                previewImg.src = e.target.result;
                previewImg.style.display = "block";
                previewImg.classList.add("animate__animated", "animate__fadeIn");
                resultDiv.innerHTML = "";
            };
            reader.readAsDataURL(file);
        }
    });

    // Predict button
    uploadBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        loadingText.style.display = "block";
        resultDiv.innerHTML = "";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            loadingText.style.display = "none";

            if (data.error) {
                resultDiv.innerHTML = `<span class="text-danger">${data.error}</span>`;
            } else {
                resultDiv.innerHTML = `
                    <span class="text-success">Prediction: ${data.class}</span><br>
                    <span class="text-secondary">Confidence: ${data.confidence}%</span>
                `;
                resultDiv.classList.add("animate__animated", "animate__fadeInUp");
            }
        } catch (err) {
            loadingText.style.display = "none";
            resultDiv.innerHTML = `<span class="text-danger">Error connecting to backend</span>`;
        }
    });

    // Clear button
    clearBtn.addEventListener("click", () => {
        fileInput.value = "";
        previewImg.src = "";
        previewImg.style.display = "none";
        resultDiv.innerHTML = "";
        loadingText.style.display = "none";
        previewImg.classList.remove("animate__animated", "animate__fadeIn");
        resultDiv.classList.remove("animate__animated", "animate__fadeInUp");
    });
});
