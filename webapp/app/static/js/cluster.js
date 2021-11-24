'use strict';

$(function () {
    const form = $('#address-form');
    const addrInput = $("#input-address");
    const prevPageBtn = $('#prev-page');
    const nextPageBtn = $('#next-page');
    const currPage = $('#page-number');
    const pagination = $('.all-pagination');
    const currResultsWindow = $('#results-window');
    const totalResults = $('.total-results');
    const resultNum = $('#result-number');
    const detailAddr = $('#detail-address');
    const detailTable = $('#detail-table');
    const resultsSection = $('.results-section');
    const anonScore = $('#anon-score');
    const anonScoreGroup = $('.anon-score-group');
    const instructions = $('#instructions');
    const invalidInput = $('#invalid-input');
    const specificResult = $('#specific-result');
    const noClusterMsg = $('#no-clusters-msg');
    const spinner = $('#spinner');
    const resultIdentifier = $('.result-identifier');
    const queryTable = $('#query-detail-table');

    let pageResults = []; //stores the results
    let firstInRange = 1;
    let queryObj = {};

    /**
     * checks if the addr is a valid ethereum address (hex and 42 char long including the 0x) 
     * @param {string} addr 
     */
    function isValid(addr) {
        const re = /[0-9A-Fa-f]{40}/g;
        addr = addr.trim();

        return ((addr.substr(0, 2) === '0x' && re.test(addr.substr(2))) //with the 0x
            || (re.test(addr))); //without the 0x
    }

    function setPagination(address, metadata) {

        const { num_pages, cluster_size, page, limit } = metadata;

        if (!cluster_size) {
            pagination.removeClass('shown');
        } else {
            pagination.addClass('shown');
        }

        //set total pages
        const totalNumPages = $("#total-num-pages");
        totalNumPages.html("&nbsp;" + num_pages);

        //set total results
        totalResults.text(cluster_size);

        //make selects
        $('#page-number > option').not(':first').remove(); //clear in case user had searched something else beforehand
        for (let p = 2; p <= num_pages; p++) {
            currPage.append('<option value=' + p + '>' + p + '</option>')
        }

        //change the actual selected currPage
        currPage.change(function () {
            queryObj.page = this.value - 1;
            queryObj.address = address;
            displayNewResults(queryObj); //backend is zero-indexed
        })

        firstInRange = page * limit + 1;

        //set results window
        currResultsWindow.text(firstInRange + '-' + Math.min(cluster_size, (page + 1) * limit))

        //set prev
        if (page > 0) {
            //there is a prev
            prevPageBtn.unbind('click').click(() => {
                queryObj.page = page - 1;
                displayNewResults(queryObj)
            });
        }

        //set next
        if (page < num_pages - 1) {
            // there is a next 
            nextPageBtn.unbind('click').click(() => {
                queryObj.page = page + 1;
                displayNewResults(queryObj)
            })
        }
        //set current:
        currPage.val(page + 1);
    }

    /**
     * query is a map. 
     * returns query string including the ? at the beginning 
     * @param {string} query 
     */
    function buildQueryString(query) {
        let queryString = '?';
        for (const [key, val] of Object.entries(query)) {
            queryString += key + '=' + val + '&';
        }
        if (queryString[queryString.length - 1] === '&') {
            queryString = queryString.substr(0, queryString.length - 1);
        }
        return queryString;
    }

    //score is represented from 0 to 1. if it's -1, then the whole anon score part will go away 
    function setAnonScore(score) {
        if (score < 0) {
            anonScoreGroup.removeClass('shown');
        } else {
            anonScore.text(parseInt(score * 100))
            anonScoreGroup.addClass('shown');
        }
    }

    function setSpecificResult() {
        specificResult.change((e) => {
            e.preventDefault();
            const query = specificResult.val();
            if (query === '') {
                delete queryObj.filter_name;
            } else {
                queryObj.filter_name = query;
            }
            delete queryObj.page; //reset to page 0 
            displayNewResults(queryObj);
        })
    }

    function setSearchOptions(schema, sort_default) {
        const sortDrop = $('#sort-dropdown');
        const filterDrop = $('#filter-dropdown');
        sortDrop.html('');
        filterDrop.html('');
        const { attribute: default_attr, descending } = sort_default;

        const attributes = Object.keys(schema);
        for (const attribute of attributes) {
            const { type, values } = schema[attribute];
            const checked = attribute === default_attr ? true : false;
            sortDrop.append(`<li class="form-check">
            <input type="radio" name="sort-by" value="${attribute}" class="form-check-input" ${checked ? 'checked' : ''} />
            <label>${attribute}</label>
            </li>`)


            if (type === 'float') {
                const [min, max] = values;
                filterDrop.append(`
                <li>
                ${attribute} range:&nbsp;<input name="filter_min_${attribute}" class="input-num filter-input" value="${min}" type="number" />
                -
                <input name="filter_max_${attribute}" class="input-num filter-input" value="${max}" type="number" }/>
                </li>
                `)
                $(`#filter_${attribute}`).slider({
                    range: true
                });
                $('.filter-input').change((e) => {
                    const elem = $(e.currentTarget);
                    const val = elem.val();
                    const name = elem.attr('name');
                    const num_val = Number(val);
                    if (val === '') {
                        delete queryObj[name]; //delete 
                    } else if (num_val < min || num_val > max) {
                        elem.val(queryObj[name]);
                        return;
                    } else {
                        queryObj[name] = val;
                    }
                    delete queryObj.page; //reset to page 0 
                    displayNewResults(queryObj)
                })
            } else if (type === 'category') {
                const div = $(document.createElement('div'));
                div.append(`<div>${attribute}:</div>`)
                const radioSelectName = attribute + '-category';
                const div2 = $(document.createElement('div')).attr('id', radioSelectName);
                for (const category of values) {
                    div2.append(`
                    <li class="form-check">
                    <input type="radio" name="filter-${attribute}" value="${category}" class="form-check-input"  />
                    <label>${category}</label>
                    </li>
                    `);
                }
                div.append(div2);
                filterDrop.append(div)
                $('#' + radioSelectName).unbind('change').change(e => {
                    const paramName = 'filter_' + attribute
                    const val = $(`input[name=filter-${attribute}]:checked`).val()
                    queryObj[paramName] = val;
                    delete queryObj.page;
                    displayNewResults(queryObj);
                })
            }

        }
        let checked = '';
        if (descending) {
            checked = 'checked';
            queryObj.descending = true;
        }
        sortDrop.append(`<input id='sort-desc' name='sort-descending' type='checkbox' ${checked}>&nbsp;descending 
        </input>
        `)
        sortDrop.unbind('change').change(e => {
            const val = $('input[name=sort-by]:checked').val()
            if (!queryObj.sort && val === default_attr || val === queryObj.sort) {
                return;
            }
            queryObj.sort = val;
            delete queryObj.page;
            displayNewResults(queryObj);
        })
        $('#sort-desc').unbind('change').change(function () {
            queryObj.descending = this.checked;
            delete queryObj.page;
            displayNewResults(queryObj);
        })

        filterDrop.append(`
        <div class="clear-filters">
        <button id="clear-filters"class="btn">clear all</button>
        </div>
        `);
        $('#clear-filters').click(() => {
            const allQueries = Object.entries(queryObj);
            for (const [key, val] of allQueries) {
                if (key.startsWith('filter_')) {
                    delete queryObj[key];
                }
            }
            displayNewResults(queryObj);
        })

        setSpecificResult();

    }

    function selectResult(e) {
        e.preventDefault();
        const elem = $(e.currentTarget);
        const i = elem.index();
        $('.result-entry').removeClass('selected');
        elem.addClass('selected');
        const deets = pageResults[i];
        const targetAddr = $(elem.children()[0]).text();

        console.assert(deets.address === targetAddr, 'address not matching: ' + deets.address + '\n' + targetAddr);
        resultNum.html(i + firstInRange);
        displayDetails(deets);
    }

    function displayDetails(item) {
        detailAddr.html(item.address);
        populateTable(detailTable, item, new Set(['id', 'address']));
    }

    function populateTable(domElem, obj, ignore=new Set()) {
        domElem.html('');
        const attributes = Object.keys(obj);

        for (const attribute of attributes) {
            if (ignore.has(attribute)) {
                continue; // don't need to display this again
            }
            const row = $(document.createElement('tr')).addClass('detail-row');

            const og_value = obj[attribute];
            const value = og_value ? og_value : "";
            row.append(`<td>${attribute}</td`);
            row.append(`<td>
                ${value}
                </td>`)
            domElem.append(row);

        }
    }

    function clearDetails() {
        detailAddr.html('');
        detailTable.html('');
    }

    function setQueryInfo(query) {
        const {address, metadata} = query;
        const combined = {...query, ...metadata};
        populateTable(queryTable, combined, new Set(['metadata', 'address', 'id', 'anonymity_score']));
    }

    /**
     * replaces existing page with new updated page of results. 
     * @param {object} query - a dict mapping from query to val. address must be a query key.
     * @param {*} firstTime  - true if this the user just entered the address or arrived on this page
     * @returns 
     */
    function displayNewResults(query = {}, firstTime = false) {
        //address must be included
        if (!query.address) {
            return;
        }

        const queryString = buildQueryString(query);
        const results = $('#results-table');
        resultsSection.addClass('shown');
        spinner.addClass('shown');
        results.html('');
        resultIdentifier.removeClass('shown');
        clearDetails();

        axios.get('/search' + queryString)
            .then(function (response) {
                spinner.removeClass('shown');
                const { success, data } = response.data;
                const { cluster, metadata, query } = data;
                anonScoreGroup.addClass('shown');
                const { schema, sort_default } = metadata;
                setPagination(query.address, metadata);
                if (firstTime) {
                    setQueryInfo(query);
                    setSearchOptions(schema, sort_default);
                }
                pageResults = []; //clear 
                if (success === 1 && cluster.length > 0) {
                    noClusterMsg.removeClass('shown');
                    resultIdentifier.addClass('shown');

                    if (firstTime) {
                        setAnonScore(query.anonymity_score);
                    }
                    for (let i = 0; i < cluster.length; i++) {
                        const address = cluster[i]
                        const entry = $(document.createElement('button')).addClass('result-entry');
                        entry.append(`<div class="entry-address">${address.address}</div>`);
                        entry.append(`<div class="entry-entity">${address.entity}</div>`);
                        results.append(entry);
                        pageResults.push(address);
                    }
                    $('.result-entry').unbind().click(selectResult);
                    $('.result-entry:first-child').click();
                } else {
                    clearDetails();
                    noClusterMsg.addClass('shown');
                    setAnonScore(1);
                    
                }
            })
            .catch(function (error) {
                results.html('<div>No clusters found.</div>');
                console.log(error);
            });
    }

    /**
     * main function!! 
     */

    function uponFirstLoad() {
        const urlParams = new URLSearchParams(window.location.search);
        const addr = urlParams.get('address');
        queryObj.address = addr;

        form.submit(e => {

            e.preventDefault();
            invalidInput.removeClass('shown');
            resultsSection.removeClass('shown');
            pagination.removeClass('shown');
            resultIdentifier.removeClass('shown');
            anonScoreGroup.removeClass('shown');
            const a = addrInput.first().val()
            instructions.removeClass('shown');
            if (!isValid(a)) {
                invalidInput.addClass('shown');
                return;
            }

            if (a !== addr) {
                window.location.href = "/cluster?address=" + a;
            } else {
                //when a new address is searched for, we need it clear the original queries. 
                queryObj = { address: a };
                displayNewResults(queryObj, true)
            }
        })
        addrInput.val(addr);
        //upon page load
        if (addr) {
            form.trigger('submit');
        }
    }
    uponFirstLoad();

    $('[data-toggle="popover"]').popover()
});

