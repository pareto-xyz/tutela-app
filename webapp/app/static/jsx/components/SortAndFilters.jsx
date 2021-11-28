import React, {useState} from 'react';
import { Dropdown, DropdownButton, ButtonGroup } from 'react-bootstrap';

export default function SortAndFilters({ refineData, getNewResults }) {
    let schema = refineData.schema || {};
    let sort_default = refineData.sort_default || {};
    const { attribute, descending } = sort_default;
    const [sortAttr, setSortAttr] = useState(attribute);
    const [desc, setDesc] = useState(descending);

    const selectSort = val => {
        setSortAttr(val);
        getNewResults(false, {page: 0, sort: val});
    }

    const getSortOption = (entry, idx) => {
        const [key, details] = entry;
        return (
            <Dropdown.Item eventKey={key} className={sortAttr === key ? 'selected-dropdown' : ''} key={idx}>{key}</Dropdown.Item>
        );
    }

    return (
        <div className="search-options">
            <Dropdown className="button-group" onSelect={selectSort}>
                <Dropdown.Toggle variant="dark"  size="sm" >
                    sort by
                </Dropdown.Toggle>

                <Dropdown.Menu>
                    {Object.entries(schema).map(getSortOption)}
                    <Dropdown.Divider />
                    <Dropdown.Item onClick={() => setDesc(!desc)} className={desc ? 'selected-dropdown' : ''}>
                        {desc && '\u2713'}
                        descending order
                        </Dropdown.Item>
                </Dropdown.Menu>
            </Dropdown>
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