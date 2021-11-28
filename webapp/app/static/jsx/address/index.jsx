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
    const { params } = props;
    const inputEl = useRef(null);

    let [queryObj, setQuery] = useState({});
    const [inputAddress, setInputAddress] = useState('');
    const [pageResults, setPageResults] = useState([]);
    const [invalid, setInvalid] = useState(false);
    const [firstView, setFirstView] = useState(true);
    const [showResultsSection, setShowResultsSection] = useState(false);
    const [loadingCluster, setLoadingCluster] = useState(false);
    const [loadingQuery, setLoadingQuery] = useState(false);
    const [queryInfo, setQueryInfo] = useState({});
    const [tornado, setTornado] = useState({});
    const [aliases, setAliases] = useState({});
    const [refineData, setRefineData] = useState({});

    const getAliases = () => {
        axios.get('/utils/aliases').then(response => {
            setAliases(response.data);
        }).catch(err => console.log(err));
    }

    // initializing
    useEffect(() => {
        getAliases();

        const addr = params.get('address');
        if (addr !== null) {
            inputEl.current.value = addr;
            setInputAddress(addr);
            submitInputAddress(addr);
        }
    }, [])

    const getNewResults = (newAddress, newQueries) => {
        if (newQueries) {
            for (const [key, value] of Object.entries(newQueries)) {
                queryObj[key] = value;
            }
            setQuery(queryObj);
        }
        setShowResultsSection(true);
        const queryString = buildQueryString(queryObj);
        setLoadingCluster(true);
        if (newAddress) {
            setLoadingQuery(true);
        }

        axios.get('/search' + queryString)
            .then(function (response) {
                setLoadingQuery(false);
                setLoadingCluster(false);
                response = example;
                const { success, data } = response;
                const { cluster, metadata, query, tornado } = data;
                if (newAddress) {
                    setRefineData(metadata);
                    setQueryInfo(query);
                    setTornado(tornado);
                }

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
        getNewResults(true);

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
                        <FormControl onKeyPress={(e) => {
                            if (e.key !== 'Enter') return;
                            e.preventDefault();
                            submitInputAddress();
                        }}
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

                    {showResultsSection && <div className="results-section">

                        <QueryInfo data={queryInfo} loading={loadingQuery} aliases={aliases} />
                        <TornadoInfo data={tornado} aliases={aliases} />

                    </div>}
                    {showResultsSection &&
                        <ClusterResults refineData={refineData}
                            results={pageResults}
                            loading={loadingCluster}
                            getNewResults={getNewResults}
                            aliases={aliases}
                        />}
                </div >
            </div>
        </div>
    )
}

export default ClusterPage;