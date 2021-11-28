import React from 'react';
import SortAndFilters from '../components/SortAndFilters';
import Pagination from '../components/Pagination';

export default function ClusterResults(props) {
    const { results} = props;
    const noResults = results.length == 0;

    return (
        <div >
            <SortAndFilters />
            <Pagination />


            <div className="results-section">
                <div className="results">
                    <div className="panel-title spaced">
                        LINKED ADDRESSES
                    </div>
                    {noResults && <div>No clusters found. </div>}
                    <table id="results-table">
                    </table>

                </div>
                <div className="detail-page">
                    <div className="result-identifier panel-sub">
                        result&nbsp;<span id="result-number">0</span>&nbsp;out of&nbsp;<span
                            className="total-results"></span>
                    </div>
                    <div id="detail-address" className="panel-title">
                        SELECTED LINKED ADDRESS
                    </div>
                    <div id="detail-table">
                    </div>
                </div>
            </div>
        </div>
    )
}