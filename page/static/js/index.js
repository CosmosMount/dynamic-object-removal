window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  $('.navbar-burger').on('click', function () {
    $('.navbar-burger').toggleClass('is-active');
    $('.navbar-menu').toggleClass('is-active');
  });

  var slidesToShow = 2;
  if (typeof window.matchMedia === 'function') {
    if (window.matchMedia('(max-width: 575px)').matches) slidesToShow = 1;
    else if (window.matchMedia('(min-width: 1200px)').matches) slidesToShow = 3;
  } else if (window.innerWidth < 576) slidesToShow = 1;

  var carouselOptions = {
    slidesToScroll: 1,
    slidesToShow: slidesToShow,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  };
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('.carousel', carouselOptions);
  }
});
