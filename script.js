const fileInput = document.getElementById("audio_file");
const fileLabelText = document.getElementById("fileLabelText");
const audioPreview = document.getElementById("audioPreview");
const audioPlayer = document.getElementById("audioPlayer");
const uploadForm = document.getElementById("uploadForm");
const loadingOverlay = document.getElementById("loadingOverlay");
const predictBtn = document.getElementById("predictBtn");
const splashScreen = document.getElementById("splashScreen");
const mainPage = document.getElementById("mainPage");

/* =========================
   AUDIO FILE PREVIEW LOGIC
   ========================= */

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];

        if (!file) {
            fileLabelText.textContent = "Choose audio file (.wav)";
            audioPreview.style.display = "none";
            audioPlayer.src = "";
            return;
        }

        fileLabelText.textContent = file.name;

        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        audioPreview.style.display = "block";
    });
}

/* =========================
   FORM SUBMISSION / LOADING
   ========================= */

if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
        if (predictBtn) {
            predictBtn.disabled = true;
        }
        if (loadingOverlay) {
            loadingOverlay.classList.remove("hidden");
        }
    });
}

/* =========================
   SPLASH SCREEN LOGIC
   ========================= */

window.addEventListener("load", () => {
    // Keep splash visible briefly, then fade it out
    setTimeout(() => {
        if (splashScreen) {
            splashScreen.classList.add("hide");
        }
        if (mainPage) {
            mainPage.setAttribute("aria-hidden", "false");
        }
    }, 2200); // duration splash is visible

    // Remove splash from DOM after fade to avoid blocking clicks
    if (splashScreen) {
        splashScreen.addEventListener(
            "transitionend",
            () => {
                try {
                    splashScreen.remove();
                } catch (e) {
                    /* ignore */
                }
            },
            { once: true }
        );
    }
});
