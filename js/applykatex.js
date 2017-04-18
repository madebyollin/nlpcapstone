(function(document) {
    var maths = document.querySelectorAll(".katex-math");
    var opts = {displayStyle: true};
    for (var i = 0; i < maths.length; i++) {
        katex.render(maths[i].textContent, maths[i], opts);
    }
}(document));
