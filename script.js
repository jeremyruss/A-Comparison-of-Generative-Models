const pane = document.querySelector(".pane");
const container = document.querySelector(".container");
const images = document.querySelector(".images");
const strip = document.getElementById("references")

container.addEventListener("mousemove", (e) => {
    let xAxis = -(container.offsetWidth / 2 - e.pageX) / 50;
    let yAxis = (container.offsetHeight / 2 - e.pageY) / 50;
    pane.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
});
container.addEventListener("mouseenter", (e) => {
    pane.style.transition = "background .5s";
    images.style.transform = "translateZ(100px)";
    pane.style.backgroundSize = "100% 100%";
});
container.addEventListener("mouseleave", (e) => {
    pane.style.transition = "all 0.5s ease";
    pane.style.transform = `rotateY(0deg) rotateX(0deg)`;
    images.style.transform = "translateZ(0px)";
    pane.style.backgroundSize = "120% 120%";
});
strip.addEventListener("mouseenter", (e) => {
    strip.style.transition = "background .5s";
    strip.style.backgroundSize = "100% 100%";
});
strip.addEventListener("mouseleave", (e) => {
    strip.style.transition = "all 0.5s ease";
    strip.style.backgroundSize = "120% 120%";
});