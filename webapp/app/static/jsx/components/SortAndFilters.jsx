import React from 'react';

export default function SortAndFilters(props) {
    return (
        <div className="search-options">
            <div className="button-group sort-group">
                <button type="button" className="btn btn-default btn-sm dropdown-toggle" data-toggle="dropdown">
                    sort
                    by<span className="caret"></span></button>
                <ul id="sort-dropdown" className="dropdown-menu">
                </ul>
            </div>
            <div className="button-group">
                <button type="button" className="btn btn-default btn-sm dropdown-toggle" data-toggle="dropdown">
                    filter
                    by<span className="caret"></span></button>
                <ul id="filter-dropdown" className="dropdown-menu">

                </ul>
            </div>
            <input id="specific-result" type="text" className="form-control"
                placeholder="search specific address or name" aria-label="ethereum address"
                aria-describedby="basic-addon2"></input>
        </div>
    );
}