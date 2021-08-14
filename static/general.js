/*
	Soul by TEMPLATE STOCK
	templatestock.co @templatestock
	Released for free under the Creative Commons Attribution 3.0 license (templated.co/license)
*/



jQuery(document).foundation();


jQuery(document).ready(function(){
    
    var currentlyScrolled;
    
    jQuery(window).scroll(function(){
        
        var header = jQuery('header');
        currentlyScrolled = jQuery(this).scrollTop();

        //Apply Sticky Header
        if( currentlyScrolled > 0){
            header.addClass('sticky');
        }
        else
        {
            header.removeClass();
        }
        
        if( jQuery(this).scrollTop() > 100)
        {
            jQuery('.backtotop').stop().animate({'right':'0'});
        }
        else
        {
            jQuery('.backtotop').stop().animate({'right':'-40px'});
        }
        
    });
    
});

jQuery(window).load(function(){
   
   //Set Portfolio Items Height
    function setWorkHeight(){
        if(jQuery('section.works ul li a figure .imgholder img').length > 0){
            
            var imageHolder = jQuery('section.works ul li a figure .imgholder');
            imageHolder.css({'height': imageHolder.find('img').height()});  
            
        }
    }
    
    setWorkHeight();


    //Services Animation
    jQuery('section.services .title p').appear();

    jQuery('body').on('appear', 'section.services .title p', function() {

            var services = [];

            jQuery('.service').each(function(){
                            services.push(jQuery(this));
            });

            for(i=0; i <= services.length; i++){
                            jQuery(services[i]).delay(i * 400).animate({'opacity':1},400);
            }

    });

    //Parallax Animation
    jQuery('section[data-type="background"]').each(function(){
            var $bgobj = $(this); // assigning the object

            jQuery(window).scroll(function() {
                            var yPos = -(jQuery(window).scrollTop() / $bgobj.data('speed')); 

                            // Put together our final background position
                            var coords = '50% '+ yPos + 'px';

                            // Move the background
                            $bgobj.css({ backgroundPosition: coords });
            }); 
    });

    //Portfolio Filtering And Animation
    var works = jQuery('ul#works');

    if(jQuery('ul#works').length > 0){
        works.isotope();

        jQuery('ul.filters').appear();

        jQuery('body').on('appear', 'ul.filters', function() {

            var works = [];

            jQuery('ul#works').find('figure').each(function(){
                             works.push(jQuery(this));
            });

            for(i=0; i <= works.length; i++){
                            jQuery(works[i]).delay(i * 400).animate({'opacity':1},400);
            }

        }); 
    }

    //Works Portfolio Show
    jQuery('section.works ul li a').click(function(){

            var getCurrentPortTop = jQuery('section.works').offset()
                    ,showPortf = jQuery(this).attr('href');

            jQuery('html, body').animate({scrollTop: getCurrentPortTop.top}, 500, function(){

                    if(jQuery('section.works article.work.slided').length > 0){
                            jQuery('section.works article.work.slided').slideUp().removeClass('slided');
                            jQuery(showPortf).slideDown(500, function(){
                                    jQuery('section.works .close').css({'display':'inline-block'});
                                    jQuery(this).addClass('slided');
                            });
                    }
                    else{
                            jQuery(showPortf).slideDown(500, function(){
                                    jQuery('section.works .close').css({'display':'inline-block'});
                                    jQuery(this).addClass('slided');
                            });
                    }

            });

       return false;

    });

    //Close Event
    jQuery('body').on('click', 'section.works .close', function(){

        var getCurrentPortTop = jQuery('section.works').offset();

        jQuery('section.works .close').css({'display':'none'});
        jQuery('section.works article.work.slided').slideUp().removeClass('slided');

        jQuery('html, body').animate({scrollTop: getCurrentPortTop.top}, 500);

        return false;
    });


    jQuery('.filters li a').click(function(){  

        if(jQuery('section.works article.work.slided').length > 0){
                jQuery('section.works article.work.slided').slideUp();
        }

        jQuery('.filters li').removeClass();
        jQuery(this).parent().addClass('selected');

        var selector = jQuery(this).attr('data-filter');
        works.isotope({ filter: selector });
        return false;

    });

    //Input Placeholders
    jQuery(".defaultText").focus(function(srcc)
      {
          if (jQuery(this).val() == jQuery(this)[0].title)
          {
              jQuery(this).removeClass("defaultTextActive");
              jQuery(this).val("");
          }
      });

      jQuery(".defaultText").blur(function()
      {
          if (jQuery(this).val() == "")
          {
              jQuery(this).addClass("defaultTextActive");
              jQuery(this).val($(this)[0].title);
          }
      });

      jQuery(".defaultText").blur(); 

    function banner(){
        //Banner Initialization
        if(jQuery('section.about .slider img').length > 1){
            
            var slider = jQuery('section.about .slider');
            
            slider.trigger("destroy");
           
            slider.carouFredSel({
                items  : {
                    visible         : 1
                },
                direction           : "left",
                
                scroll : {
                    items           : 1,
                    fx              : "crossfade",
                    easing          : "linear",
                    duration        : 1000,                         
                    pauseOnHover    : true
                },
                prev : {
                    button          : 'section.about .sliderHolder .bannerNav a.prev'
                },
                next : {
                    button          : 'section.about .sliderHolder .bannerNav a.next'
                }, 
                auto : { 
                    play            : true
                },
                responsive          : true
            });
            
        }
        else
        {
            jQuery('section.about .sliderHolder .bannerNav a').css({'display':'none'});  
        }
    }
    
    banner();
    
    //Services Height
    function setServiceHeight(){
        
        var serviceHeight = 0;
        
        if(jQuery('section.services .service').length > 0){
            
            jQuery('section.services .service').css({'height':'auto'});
            
            jQuery('section.services .service').each(function(){
                
                if(jQuery(this).height() > serviceHeight ){
                    serviceHeight = jQuery(this).height();
                }
            });
            
            jQuery('section.services .service').css({'height':serviceHeight + 20});
            
        }
    }
    
    setServiceHeight();
     
	function testimonialsSlider(){
	
            var testimonialsSlider = jQuery('section.testimonials .holder');

            if(testimonialsSlider.find('article').length > 1){

                testimonialsSlider.trigger("destroy");

                testimonialsSlider.find('article').css({'width':'auto', 'height':'auto'});

                var testimonialSliderWidth = testimonialsSlider.find('article').eq(0).width();
                var testimonialSliderHeight = testimonialsSlider.find('article').eq(0).height();

                testimonialsSlider.find('article').css({'width':testimonialSliderWidth, 'height':testimonialSliderHeight});

                testimonialsSlider.carouFredSel({
                        items  : {
                                visible         : 1, 
                                width           : testimonialSliderWidth,
                                height          : testimonialSliderHeight
                        },
                        direction           : "left",
                        scroll : {
                                items           : 1,
                                fx              : "crossfade",
                                easing          : "linear",
                                duration        : 800,                         
                                pauseOnHover    : true
                        },
                        prev : {
                                button          : 'section.testimonials .bannerNav a.prev'
                        },
                        next : {
                                button          : 'section.testimonials .bannerNav a.next'
                        }, 
                        auto : { 
                                play            : true
                        }
                });
            }
    }
	
    testimonialsSlider();
    
      
      function memberSlider(){
        //Member Slider
        if(jQuery('section.team .slider .sliderholder .member').length > 0){
            
            jQuery('section.team .slider .sliderholder').trigger("destroy");
            jQuery('section.team .slider .sliderholder .member').css({'width':'auto', 'height':'auto'});

            var memberSliderWidth = jQuery('section.team .slider .sliderholder .member').eq(0).width();
            var memberSliderHeight = jQuery('section.team .slider .sliderholder .member').eq(0).height();

            jQuery('section.team .slider .sliderholder .member').css({'width':memberSliderWidth, 'height':memberSliderHeight});

            jQuery('section.team .slider .sliderholder').carouFredSel({

                items  : {
                  visible         : 1,
                  width           : memberSliderWidth,
                  height          : memberSliderHeight
              },

              direction           : "left",

              scroll : {
                  items           : 1,
                  fx              : "crossfade",
                  easing          : "linear",
                  duration        : 800,                         
                  pauseOnHover    : true
              },
              prev : {
                  button          : 'section.team .slider .bannerNav a.prev'
              },
              next : {
                  button          : 'section.team .slider .bannerNav a.next'
              }, 
              auto : { 
                  play            : false
              },
              responsive          : true
          });

        }
      }
      
      memberSlider();
      
      //Page Scroll
      jQuery(".scroll").click(function(event) {

        //prevent the default action for the click event
        event.preventDefault();
        //get the full url - like mysitecom/index.htm#home
        var full_url = this.href;
        //split the url by # and get the anchor target name - home in mysitecom/index.htm#home

        var parts = full_url.split("#");
        var trgt = parts[1];
        //get the top offset of the target anchor

        var target_offset = jQuery("#" + trgt).offset();

        var target_top = target_offset.top - 59;

        //goto that anchor by setting the body scroll top to anchor top
        jQuery('html, body').animate({scrollTop: target_top}, 1000);

       });
       
        //On Window Resize
        jQuery(window).resize(function(){
            setWorkHeight();
            testimonialsSlider();
            setServiceHeight();
            works.isotope();
            memberSlider();
            banner();
       });


});