$(document).ready(function() {
  $('#devicedropdownid').hide();
    var testSelect = document.getElementById('testdropdownid');
    var testValue = testSelect.options[testSelect.selectedIndex].value;

    var deviceSelect = document.getElementById('devicedropdownid');
    var deviceValue = deviceSelect.options[deviceSelect.selectedIndex].value;
   
  $('input:radio[name="test"]').change(
function() {
	if ($(this).is(':checked') )
	{$('.uklist').show();
     }
	// else if ($(this).is(':checked') && $(this).val() == 'es')
	// {
	// 	$('.uklist').hide();
  //   $('.delist').hide();
  //   $('.nllist').hide();
  //   $('.frlist').hide();
  //   $('.eslist').show();
	// 	}
  // // FINISH FROM HERE
  // else if ($(this).is(':checked') && $(this).val() == 'de')
	// {
	// 	$('.uklist').hide();
  //   $('.delist').show();
  //   $('.nllist').hide();
  //   $('.frlist').hide();
  //   $('.eslist').hide();
	// 	}
  
  // else if ($(this).is(':checked') && $(this).val() == 'fr')
	// {
	// 	$('.uklist').hide();
  //   $('.delist').hide();
  //   $('.nllist').hide();
  //   $('.frlist').show();
  //   $('.eslist').hide();
	// 	}
  
  
  else {
    $('.uklist').show();
    // $('.delist').hide();
    // $('.nllist').show();
    // $('.frlist').hide();
    // $('.eslist').hide();
   }
  
	}
);

  
  
}
);