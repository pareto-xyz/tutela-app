import React, { useState, useRef, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { Form, FormControl, InputGroup } from 'react-bootstrap';
import { isValid, buildQueryString } from '../components/utils';
import axios from 'axios';
import example from '../../data/example';
import QueryInfo from './QueryInfo';
import TornadoInfo from './TornadoInfo';
import schemaResponse from '../../data/schema';
import TpoolOverall from './TpoolOverall';
import TpoolStats from './TpoolStats';
import { QueryObjContext } from '../components/Contexts';
import SortAndFilters from '../components/SortAndFilters';
import Pagination from '../components/Pagination';
import aliasesResponse from '../../data/aliases';

import AccordionOfResults from '../components/AccordionOfResults';
import HaveIBeenCompromised from './HaveIBeenCompromised';


//to be displayed instead of listed cluster, if no clusters were found. 
const NoClusters = (
    <div>
        <div className="center-inside">
            No clusters found.
            <br />
            <img width="200" src="/static/img/spy.png" />
            <br />
            <div>you're anonymous, harry!</div>
        </div>
    </div>
)

const AddressClusterHeader = (
    <div className="results">
        <div className="panel-title">
            LINKED ADDRESSES
        </div>
    </div>
)

function ClusterPage(props) {
    const { params } = props;
    const inputEl = useRef(null);

    let [queryObj, setQuery] = useState({}); //sorts and filters 
    const [inputAddress, setInputAddress] = useState('');
    const [pageResults, setPageResults] = useState([]);
    const [invalid, setInvalid] = useState(false);
    const [firstView, setFirstView] = useState(true);
    const [showResultsSection, setShowResultsSection] = useState(false);
    const [loadingCluster, setLoadingCluster] = useState(false);
    const [loadingQuery, setLoadingQuery] = useState(false);
    const [queryInfo, setQueryInfo] = useState({}); //info about the input 
    const [tornado, setTornado] = useState({});
    const [aliases, setAliases] = useState({});
    const [schema, setSchema] = useState({});
    const [paginationData, setPaginationData] = useState({});
    const [searchType, setSearchType] = useState(null);

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
    }, []);

    const setSort = attrObj => {
        getNewResults(false, { page: 0, ...attrObj });
    }

    const getNewResults = (newAddress, newQueries) => {
        if (newQueries === 'clear' || newAddress) {
            const allQueries = Object.keys(queryObj);
            for (const key of allQueries) {
                if (key.startsWith('filter_')) {
                    delete queryObj[key];
                }
            }
            setQuery(queryObj);
        } else if (newQueries) { //assume it's an object. 
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
                // response = schemaResponse;
                const { success, data, is_tornado } = response.data;
                if (is_tornado === 1) {
                    const { query } = data;
                    setQueryInfo(query);
                    setSearchType('tornadoPool');
                } else if (is_tornado === 0) {
                    const { cluster, metadata, query, tornado } = data;
                    const { cluster_size, limit, num_pages, page } = metadata;
                    setPaginationData({ total: cluster_size, limit, num_pages, page });
                    const { sort_default } = metadata;
                    const { attribute, descending } = sort_default;
                    if (newAddress) {
                        queryObj.sort = attribute;
                        queryObj.descending = descending;
                        setQuery(queryObj);
                        setSchema(metadata.schema);
                        setQueryInfo(query);
                        setTornado(tornado);
                    }
                    setSearchType('other')

                    if (success === 1 && cluster.length > 0) {
                        setPageResults(cluster);
                    } else {
                        setPageResults([]);
                    }
                }

            })
            .catch(function (error) {
                setPageResults([]);
                setLoadingQuery(false);
                setLoadingCluster(false);
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
        queryObj = {
            address: addr
        }; // clears all the sort, filters, page, etc.
        setQuery(queryObj);
        getNewResults(true);
        window.history.replaceState(null, null, "?address=" + addr);

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

                    {searchType === 'tornadoPool' &&
                        <>
                            {showResultsSection &&
                                <div>
                                    <div className="tornado-results-section ">

                                        <TpoolOverall data={queryInfo} loading={loadingQuery} />
                                        {queryInfo.metadata && <TpoolStats data={queryInfo.metadata.stats} aliases={aliases} />}

                                    </div>
                                    <HaveIBeenCompromised aliases={aliases} tcashAddr={inputAddress} />
                                </div>
                            }

                        </>}
                    {searchType === 'other' && <>
                        {showResultsSection && <div className="results-section">

                            <QueryInfo data={queryInfo} loading={loadingQuery} aliases={aliases} />
                            <TornadoInfo data={tornado} aliases={aliases} />

                        </div>}
                        {showResultsSection &&
                            <QueryObjContext.Provider value={queryObj}>

                                <AccordionOfResults
                                    sectionHeader={AddressClusterHeader}
                                    rowTitle='address'
                                    rowBadge='entity'
                                    Pagination={<Pagination paginationData={paginationData} getNewResults={getNewResults} />}
                                    results={pageResults}
                                    loading={loadingCluster}
                                    aliases={aliases}
                                    noDataComponent={NoClusters}
                                    SortAndFilters={<SortAndFilters schema={schema} setSort={setSort} getNewResults={getNewResults} />}
                                />
                            </QueryObjContext.Provider>

                        }
                    </>}

                </div >

            </div>
            <Footer />

        </div>
    )
}

export default ClusterPage;