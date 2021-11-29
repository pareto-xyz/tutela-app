import React, { useState } from 'react';
import { Dropdown, Accordion, Form, InputGroup, FormControl, } from 'react-bootstrap';

const CHECKMARK = '\u2713';


export default function SortAndFilters({ schema, descendingSort, sortBy, setSort, getNewResults }) {

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
        return (
            <Dropdown.Item eventKey={key} className={sortBy === key ? 'selected-dropdown' : ''} key={idx}>{key}</Dropdown.Item>
        );
    }

    const FloatBody = ({ k, values }) => {
        const [min, max] = values;
        const [minInvalid, setMinInvalid] = useState(false);
        const [maxInvalid, setMaxInvalid] = useState(false);
        const [minVal, setMinVal] = useState(min);
        const [maxVal, setMaxVal] = useState(max);

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
            const key = `filter_${minOrMax}_${k}`;
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

    const CategoryBody = ({ k, values }) => {
        const [selected, setSelected] = useState(null);
        return (
            <Accordion.Body className="dropdown-accordion">
                {values.map((value, idx) =>
                    <Dropdown.Item className={selected === value ? 'selected-dropdown' : ''}
                        key={idx}
                        eventKey={value}
                        onClick={e => {
                            setSelected(value);
                            let obj = { page: 0 };
                            obj['filter_' + k] = value;
                            getNewResults(false, obj);
                        }}>
                        {value}
                    </Dropdown.Item>)}
            </Accordion.Body>
        )
    }

    const FilterOption = ({ entry, idx }) => {
        const [key, details] = entry;
        const [selected, setSelected] = useState(false);

        if (key === 'name' || key === 'address') {
            return <></>; //ignore these
        }

        const { type, values } = details;

        return (
            <Accordion.Item eventKey={idx} className='dropdown-accordion' key={idx}>
                <Accordion.Header className="dropdown-accordion" onClick={() => setSelected(!selected)}>
                    <div>{key}</div> <div className="expand-symbol">{selected ? '-' : '+'}</div>
                </Accordion.Header>
                {type === 'float' && <FloatBody k={key} values={values} />}
                {type === 'category' && <CategoryBody k={key} values={values} />}
            </Accordion.Item>
        );
    }

    const FilterByName = () => {
        const [val, setVal] = useState('');
        return (
            <InputGroup  >
                <FormControl className="specific-result"
                    value={val}
                    placeholder="search specific address or name"
                    onChange={(e) => setVal(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key !== 'Enter') return;
                        e.preventDefault();
                        getNewResults(false, {page: 0, filter_name: val});
                    }}
                />
            </InputGroup>
        );
    }

    return (
        <div className="search-options">
            <Dropdown className="button-group" onSelect={selectSort}>
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
            <Dropdown className="button-group">
                <Dropdown.Toggle variant="dark" size="sm" >
                    filter by
                </Dropdown.Toggle>

                <Dropdown.Menu>
                    <Accordion>
                        {Object.entries(schema).map((entry, idx) => <FilterOption idx={idx} key={idx} entry={entry} />)}
                    </Accordion>
                    <Dropdown.Divider />
                    <Dropdown.Item className="flush-right" onClick={() => getNewResults(false, 'clear')} >
                        (clear all)
                    </Dropdown.Item>
                </Dropdown.Menu>

            </Dropdown>

            <FilterByName />

        </div>
    );
}