const formData = new FormData();
formData.append("image", selectedImage);

try {
    const response = await fetch("/caption", {
        method: "POST",
        body: formData
    });

    const text = await response.text();

    let data;
    try {
        data = JSON.parse(text);
    } catch {
        throw new Error(`Server returned ${response.status}`);
    }

    if (!response.ok) {
        throw new Error(data.error || "Caption generation failed.");
    }

    captionBox.innerText = data.caption;

} catch (err) {
    captionBox.innerText = err.message;
}