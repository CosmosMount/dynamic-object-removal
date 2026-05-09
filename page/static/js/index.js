window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  $('.navbar-burger').on('click', function () {
    $('.navbar-burger').toggleClass('is-active');
    $('.navbar-menu').toggleClass('is-active');
  });

  var carouselOptions = {
    slidesToScroll: 1,
    slidesToShow: 2,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  };
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('.carousel', carouselOptions);
  }
});
