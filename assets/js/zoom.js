// Initialize medium zoom.
$(document).ready(function() {
  medium_zoom = mediumZoom('[data-zoomable], .post-content p > img', {
    background: getComputedStyle(document.documentElement)
        .getPropertyValue('--global-bg-color') + 'ee',  // + 'ee' for trasparency.
  })

  // Preserve the legibility of panoramic, label-dense diagrams in print. CSS
  // cannot query an image's intrinsic aspect ratio, so annotate the containing
  // paragraph after the asset is available and let the print stylesheet assign
  // that figure to a landscape page.
  document.querySelectorAll('.post-content p > img').forEach(function(image) {
    const annotatePrintLayout = function() {
      if (image.naturalWidth / image.naturalHeight >= 2.1) {
        image.closest('p')?.classList.add('post-figure--print-landscape')
      }
    }
    if (image.complete) annotatePrintLayout()
    else image.addEventListener('load', annotatePrintLayout, { once: true })
  })
});
