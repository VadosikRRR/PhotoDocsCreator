document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    if (!file) {
        alert('Пожалуйста, выберите файл');
        return;
    }
    
    loadingIndicator.style.display = 'flex';
    
    const originalImage = document.getElementById('originalImage');
    const originalUrl = URL.createObjectURL(file);
    originalImage.src = originalUrl;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Ошибка обработки изображения');
        }
        
        const blob = await response.blob();
        const processedUrl = URL.createObjectURL(blob);
        const processedImage = document.getElementById('processedImage');
        processedImage.src = processedUrl;
        
        const downloadLink = document.getElementById('downloadLink');
        downloadLink.href = processedUrl;
        downloadLink.download = 'processed_image.png';
        
        document.getElementById('imagesContainer').style.display = 'flex';
    } catch (error) {
        console.error('Ошибка:', error);
        alert('Произошла ошибка при обработке изображения');
    } finally {
        loadingIndicator.style.display = 'none';
    }
});