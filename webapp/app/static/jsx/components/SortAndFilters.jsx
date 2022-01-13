import React, { useState, useEffect, useContext, useRef } from 'react';
import { Dropdown, Accordion, Form, InputGroup, FormControl, } from 'react-bootstrap';
import { QueryObjContext } from './Contexts';

const CHECKMARK = '\u2713';

/**
 * FloatBody - allows user to filter via range of something 
 * @param {*} k: name of that filter option, values: min and max option, getNewResult: rerenders page 
 * @returns 
 */
const FloatBody = ({ k, values, getNewResults }) => {
    const getFilterParam = (minOrMax, k) => `filter_${minOrMax}_${k}`

    const [min, max] = values;
    const [minInvalid, setMinInvalid] = useState(false);
    const [maxInvalid, setMaxInvalid] = useState(false);
    const [minVal, setMinVal] = useState(min);
    const [maxVal, setMaxVal] = useState(max);

    const queryObjContext = useContext(QueryObjContext);
    const filterMin = queryObjContext[getFilterParam('min', k)];
    const filterMax = queryObjContext[getFilterParam('max', k)];

    //in case filters were cleared 
    useEffect(() => {
        if (filterMin === undefined && filterMax === undefined) {
            setMinVal(min);
            setMaxVal(max);
        }
    }, [filterMin, filterMax])


    const submit = (minOrMax, val) => {
        if (minOrMax === 'min') {
            if (val < min) {
                setMinInvalid(true);
                return;
            } else if (minInvalid) {
                setMinInvalid(false);
            }
        }
        if (minOrMax === 'max') {
            if (val > max) {
                setMaxInvalid(true);
                return;
            } else if (maxInvalid) {
                setMaxInvalid(false);
            }
        }
        const key = getFilterParam(minOrMax, k);
        let obj = { page: 0 };
        obj[key] = val;
        getNewResults(false, obj);
    }
    return (
        <Accordion.Body className="dropdown-accordion">
            <InputGroup hasValidation className="my-input" >
                <InputGroup.Text >min</InputGroup.Text>
                <FormControl
                    value={minVal}
                    isInvalid={minInvalid}
                    onChange={(e) => setMinVal(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key !== 'Enter') return;
                        e.preventDefault();
                        submit('min', minVal);
                    }}
                />
                <Form.Control.Feedback type="invalid">
                    Value should be no less than {min}.
                </Form.Control.Feedback>
            </InputGroup>
            <InputGroup hasValidation className="my-input" onSubmit={val => submit('max', val)}>
                <InputGroup.Text >max</InputGroup.Text>
                <FormControl
                    value={maxVal}
                    isInvalid={maxInvalid}
                    onChange={e => setMaxVal(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key !== 'Enter') return;
                        e.preventDefault();
                        submit('max', maxVal);
                    }}
                />
                <Form.Control.Feedback type="invalid">
                    Value should be no greater than {max}.
                </Form.Control.Feedback>
            </InputGroup>
        </Accordion.Body>
    );
}

/**
 * CategoryBody - filter option for categorical filter 
 * @param {*} k: name of that filter option, values: category options, getNewResults: rerenders
 * @returns 
 */
const CategoryBody = ({ k, values, getNewResults }) => {

    const getFilterParam = categoryName => 'filter_' + categoryName; //example: filter_entity 

    //check whether the response used this filter. 
    const filter = useContext(QueryObjContext)[getFilterParam(k)];

    return (
        <Accordion.Body className="dropdown-accordion">
            {values.map((value, idx) =>
                <Dropdown.Item className={filter === value ? 'selected-dropdown' : ''}
                    key={idx}
                    eventKey={value}
                    onClick={e => {
                        let obj = { page: 0 };
                        obj[getFilterParam(k)] = value;
                        getNewResults(false, obj);
                    }}>
                    {value}
                </Dropdown.Item>)}
        </Accordion.Body>
    )
}

const FilterByName = ({ ogVal, getNewResults }) => {
    const [val, setVal] = useState(ogVal || '');
    const inputEl = useRef(null);

    useEffect(() => {
        //in case filters were cleared 
        if (ogVal === undefined) {
            setVal('');
            inputEl.current.value = '';
        }
    }, [ogVal])

    return (
        <InputGroup className="col-sm-12 col-md-12 col-lg-4 link-input" >
            <FormControl className="specific-result"
                placeholder="search specific address" // or name"
                onChange={(e) => {
                    e.preventDefault();
                    setVal(e.target.value);
                }}
                onKeyPress={(e) => {
                    if (e.key !== 'Enter') {
                        return;
                    }
                    getNewResults(false, { page: 0, filter_name: val });
                }}
                ref={inputEl}
            />
        </InputGroup>
    );
}


/**
 * FilterOption - these go into the filter dropdown 
 * @param {*} entry: name of filter option, idx: index of filter option, getNewResults: rerenders everything 
 * @returns 
 */
const FilterOption = ({ entry, idx, getNewResults }) => {
    const [key, details] = entry;
    const [selected, setSelected] = useState(false);

    if (key === 'name' || key === 'address') {
        return <></>; //ignore these
    }

    const { type, values } = details;

    return (
        <Accordion.Item eventKey={idx} className='dropdown-accordion' key={idx}>
            <Accordion.Header className="dropdown-accordion" onClick={() => setSelected(!selected)}>
                <div>{key === 'conf' ? 'confidence' : key}</div> <div className="expand-symbol">&#x25BC;</div>
            </Accordion.Header>
            {type === 'float' && <FloatBody k={key} values={values} getNewResults={getNewResults} />}
            {type === 'category' && <CategoryBody k={key} values={values} getNewResults={getNewResults} />}
        </Accordion.Item>
    );
}

export default function SortAndFilters({ schema, setSort, getNewResults }) {

    const { descending: descendingSort, filter_name } = useContext(QueryObjContext);

    const selectSort = val => {
        if (val === 'descending') {
            return; //ignore
        }
        setSort({ sort: val });
    }

    const setDesc = desc => {
        setSort({ descending: desc });
    }


    const getSortOption = (key, idx) => {
        const sortBy = useContext(QueryObjContext).sort;
        return (
            <Dropdown.Item eventKey={key} className={sortBy === key ? 'selected-dropdown' : ''} key={idx}>{key === 'conf' ? 'confidence' : key}</Dropdown.Item>
        );
    }

    return (
        <div className="search-options col-12">
            <div className="row">
                <Dropdown className="col-sm-12 col-md-6 col-lg-4 button-group" onSelect={selectSort}>
                    <Dropdown.Toggle variant="dark" size="sm" >
                        sort by
                    </Dropdown.Toggle>

                    <Dropdown.Menu>
                        {Object.keys(schema).map(getSortOption)}
                        <Dropdown.Divider />
                        <Dropdown.Item onClick={() => setDesc(!descendingSort)} eventKey='descending' className={descendingSort ? 'selected-dropdown' : ''}>
                            {descendingSort && CHECKMARK}
                            descending order
                        </Dropdown.Item>
                    </Dropdown.Menu>
                </Dropdown>
                <Dropdown className="col-sm-12 col-md-6 col-lg-4 button-group">
                    <Dropdown.Toggle variant="dark" size="sm" >
                        filter by
                    </Dropdown.Toggle>

                    <Dropdown.Menu>
                        <Accordion>
                            {Object.entries(schema).map((entry, idx) => <FilterOption idx={idx} key={idx} entry={entry} getNewResults={getNewResults} />)}
                        </Accordion>
                        <Dropdown.Divider />
                        <Dropdown.Item className="flush-right" onClick={() => getNewResults(false, 'clear')} >
                            (clear all)
                        </Dropdown.Item>
                    </Dropdown.Menu>

                </Dropdown>

                <FilterByName ogVal={filter_name} getNewResults={getNewResults} />
            </div>
        </div>
    );
}