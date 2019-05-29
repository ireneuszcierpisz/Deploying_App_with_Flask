// https://www.wikitechy.com/tutorials/javascript/preview-an-image-before-it-is-uploaded
// Uses URL.createObjectURL() on the file from <input> to pass this URL to img.src
// to the browser to load the provided image

var loadFile = function(event) {
        var image = document.getElementById('flower');
        image.scr = URL.createObjectURL(event.target.files[0]);
};
