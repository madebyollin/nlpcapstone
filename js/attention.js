(function() {
    "use strict";

    function request(url, callback) {
        var xhr = new XMLHttpRequest();
        xhr.onload = e => callback(JSON.parse(xhr.response));
        xhr.open("GET", url);
        xhr.send();
    }

    var colors = {
        ANGER:   [255, 32,  88],  // Red
        SADNESS: [59,  163, 252], // Blue
        JOY:     [171, 216, 0]    // Light green
    }

    function rgba(rgb, alpha) {
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
    }

    var key = document.getElementById("attention-key");
    var colorNames = Object.keys(colors);
    for (let i = 0; i < colorNames.length; i++) {
        let label = document.createElement("span");
        label.textContent = colorNames[i];
        label.style.backgroundColor = rgba(colors[colorNames[i]], 1.0);
        if (i > 0) {
            key.appendChild(document.createTextNode(" / "));
        }
        key.appendChild(label);
    }

    var attentions = document.querySelectorAll(".attention");
    for (let parent of attentions) {
        request(parent.dataset.filepath, obj => {
            for (let k = 0; k < obj.words.length; k++) {
                let span = document.createElement("span");
                span.textContent = obj.words[k];
                span.style.backgroundColor = rgba(colors[obj.pred_label], obj.attention[k]);
                span.className = "token";
                parent.appendChild(span);
                parent.appendChild(document.createTextNode(" "));
            }
        });
    }
}());