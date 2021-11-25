import React, { useState } from 'react';
import Header from '../components/Header';
import { Form, FormControl, InputGroup } from 'react-bootstrap';
import { isValid, buildQueryString } from '../components/utils';
import axios from 'axios';
import example from '../../data/example'

import ClusterResults from './ClusterResults';

function ClusterPage() {
    const [queryObj, setQuery] = useState({});
    const [inputAddress, setInputAddress] = useState('');
    const [pageResults, setPageResults] = useState([]);
    const [firstInRange, setFirstInRange] = useState(1);
    const [invalid, setInvalid] = useState(false);
    const [firstView, setFirstView] = useState(true);
    const [showResultsSection, setShowResultsSection] = useState(false);
    const [loading, setLoading] = useState(false);

    const getNewResults = _ => {
        setShowResultsSection(true);
        const queryString = buildQueryString(queryObj);
        setLoading(true);

        axios.get('/search' + queryString)
            .then(function (response) {
                console.log(response.data);
                setLoading(false);
                const { success, data } = response.data;
                const { cluster, metadata, query, tornado } = data;
                const { schema, sort_default } = metadata;
                setPagination(query.address, metadata);
                if (firstTime) {
                    setQueryInfo(query);
                    setTornadoInfo(tornado);
                    setSearchOptions(schema, sort_default);
                }
                setPageResults(cluster);
                if (success === 1 && cluster.length > 0) {
                    noClusterMsg.removeClass('shown');
                    resultIdentifier.addClass('shown');

                    if (firstTime) {
                        setAnonScore(query.anonymity_score);
                    }


                } else {
                    clearDetails();
                    noClusterMsg.addClass('shown');
                    setAnonScore(1);

                }
            })
            .catch(function (error) {
                setPageResults([]);
                setLoading(false);
                console.log(error);
            });
    }

    const submitInputAddress = e => {
        e.preventDefault();
        if (!isValid(inputAddress)) {
            setInvalid(true);
            return;
        }
        if (firstView) {
            setFirstView(false);
        }
        setInvalid(false);
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
                        <FormControl onKeyPress={(e) => e.key === 'Enter' && submitInputAddress(e)}
                            onChange={onChangeInputAddress}
                            placeholder='eg. 0x000000000000000..........'
                            className="search-bar"
                            isInvalid={invalid}
                        >
                        </FormControl>

                        <InputGroup.Text className="right-submit-icon"><img width="15" src="/static/img/loupe.svg" alt="search"></img> </InputGroup.Text>
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

                        <div className="query-info ">
                            <div className="panel-sub">about your input</div>
                            <div className="panel-title">
                                OVERALL INFO
                            </div>


                            <div className="anon-score-group">
                                anonymity score: &nbsp;<span id="anon-score"></span> &nbsp;/ 100
                                <div data-container="body" data-toggle="popover" data-placement="bottom" data-trigger="hover"
                                    data-content="The higher the anonymity score, the less we believe this address or transaction has revealed about its privacy. Number of reveals, the connectedness of addresses and the types of reveal affect this."
                                    className="help-circle">?</div>
                            </div>
                            {loading && <div id="spinner" className="justify-content-center">
                                <div className="spinner-border" role="status">
                                    <span className="sr-only">Loading...</span>
                                </div>
                            </div>}
                            <div id="query-detail-table" className="detail-table">
                            </div>

                        </div>

                        <div className="tornado-info ">
                            <div className="panel-sub">tornado cash statistics</div>
                            <div className="panel-title">
                                TORNADO CASH INFO
                            </div>
                            <div className="panel-sub">
                                Compromised addresses are deposit addresses that are deemed revealing of identity through Tutela heuristics.
                            </div>
                            <div id="tornado-detail-table" className="detail-table">
                            </div>
                        </div>


                    </div>}
                    {showResultsSection &&
                        <ClusterResults results={pageResults} />}
                </div >
            </div>
        </div>
    )
}

export default ClusterPage;