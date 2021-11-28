import React, { useState, useRef, useEffect } from 'react';
import Header from '../components/Header';
import { Form, FormControl, InputGroup } from 'react-bootstrap';
import { isValid, buildQueryString } from '../components/utils';
import axios from 'axios';
import example from '../../data/example';
import QueryInfo from './QueryInfo';
import TornadoInfo from './TornadoInfo';

import ClusterResults from './ClusterResults';

function ClusterPage(props) {
    const {params} = props;
    const inputEl = useRef(null);

<<<<<<< HEAD
    let [queryObj, setQuery] = useState({});
=======
    const [queryObj, setQuery] = useState({});
>>>>>>> dbdef9f (support address in the param)
    const [inputAddress, setInputAddress] = useState('');
    const [pageResults, setPageResults] = useState([]);
    const [invalid, setInvalid] = useState(false);
    const [firstView, setFirstView] = useState(true);
    const [showResultsSection, setShowResultsSection] = useState(false);
    const [loading, setLoading] = useState(false);
    const [queryInfo, setQueryInfo] = useState({});
    const [tornado, setTornado] = useState({});

    // initializing
    useEffect(() => {
        const addr = params.get('address');
        if (addr !== null) {
            console.log(inputEl.current);
            inputEl.current.value = addr;
            // inputEl.current.submit();
            setInputAddress(addr);
            submitInputAddress(addr);
        }
    }, [])

    // initializing
    useEffect(() => {
        const addr = params.get('address');
        if (addr !== null) {
            console.log(inputEl.current);
            inputEl.current.value = addr;
            // inputEl.current.submit();
            setInputAddress(addr);
            submitInputAddress(addr);
        }
    }, [])

    const getNewResults = _ => {
        setShowResultsSection(true);
        const queryString = buildQueryString(queryObj);
        setLoading(true);

        axios.get('/search' + queryString)
            .then(function (response) {
                response = example;

                setLoading(false);
                const { success, data } = response;
                const { cluster, metadata, query, tornado } = data;
                const { schema, sort_default } = metadata;
                setQueryInfo(query);
                console.log(query);
                setTornado(tornado);

                if (success === 1 && cluster.length > 0) {
                    setPageResults(cluster);
                    
                } else {
                }
            })
            .catch(function (error) {
                setPageResults([]);
                setLoading(false);
                console.log(error);
            }).finally(() => {
                if (firstView) {
                    setFirstView(false);
                }
            });
    }

    const submitInputAddress = addr => {
        //param is optional. otherwise, use the inputAddress var
        if (!addr) {
            addr = inputAddress;
            setInputAddress(addr);
        }
        if (!isValid(addr)) {
            setInvalid(true);
            return;
        }
        setInvalid(false);
        queryObj.address = addr;
        setQuery(queryObj);
        getNewResults();

    }

    const onChangeInputAddress = e => {
        e.preventDefault();
        setInputAddress(e.target.value);
    }

    return (
        <div>
            <Header current={'address'} />

            <div className="container halved-bar">
                <div>
                    {firstView && <div id="instructions">
                        Enter an ethereum address to see likely connected ethereum addresses (ie. its cluster)
                        based on public data on previous transactions.
                    </div>}
                    <InputGroup onSubmit={submitInputAddress} className="mb-3 " hasValidation>
<<<<<<< HEAD
                        <FormControl onKeyPress={(e) => {
                            if (e.key !== 'Enter') return; 
                            e.preventDefault();
                            submitInputAddress();
                        }}
=======
                        <FormControl onKeyPress={(e) => e.key === 'Enter' && e.preventDefault() && submitInputAddress()}
>>>>>>> dbdef9f (support address in the param)
                            onChange={onChangeInputAddress}
                            placeholder='eg. 0x000000000000000..........'
                            className="search-bar"
                            isInvalid={invalid}
                            ref={inputEl}
                        >
                        </FormControl>

                        <InputGroup.Text onClick={submitInputAddress} className="right-submit-icon"><img width="15" src="/static/img/loupe.svg" alt="search"></img> </InputGroup.Text>
                        <Form.Control.Feedback type="invalid">
                            Please enter a valid ethereum address.
                        </Form.Control.Feedback>
                    </InputGroup>

                    {/* <div id="address-form">
                        <form className="input-group search-bar">
                            <input id="input-address" type="text" className="form-control"
                                placeholder="eg. 0x000000000000000.........." aria-label="ethereum address"
                                aria-describedby="basic-addon2"></input>
                            <button className="btn" type="submit"><img width="20" src="/static/img/loupe.svg" alt="search"></img></button>
                        </form>
                    </div> */}
                    {/* {invalid && <div >
                        Please enter a valid ethereum address.
                    </div>} */}
                    {showResultsSection && <div className="results-section">

                        <QueryInfo data={queryInfo} loading={loading} />

                        <TornadoInfo data={tornado} />

                    </div>}
                    {showResultsSection &&
                        <ClusterResults results={pageResults} loading={loading}/>}
                </div >
            </div>
        </div>
    )
}

export default ClusterPage;