'use strict';


$(function () {

    var form = $('#address-form');
    form.submit(e => {
        e.preventDefault();
        const a = $( "#input-address" ).first().val()
        window.location.href = "/cluster?address=" + a;
    })
});
