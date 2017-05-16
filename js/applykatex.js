/*global katex */

(function(document) {
    var maths, opts;

    // "Deprecated"
    maths = document.querySelectorAll(".katex-math");
    opts = {displayMode: true};
    for (var i = 0; i < maths.length; i++) {
        katex.render(maths[i].textContent, maths[i], opts);
    }


    // Use these two going forward!
    maths = document.querySelectorAll(".math-display");
    opts = {displayMode: true};
    for (var i = 0; i < maths.length; i++) {
        katex.render(maths[i].textContent, maths[i], opts);
    }

    maths = document.querySelectorAll(".math-inline");
    for (var i = 0; i < maths.length; i++) {
        katex.render(maths[i].textContent, maths[i]);
    }


}(document));
