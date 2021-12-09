import React from 'react';

export default function TpoolOverall({data, loading}) {
    const {metadata} = data;
    const {amount, currency, stats} = metadata || {};
    let {num_deposits, tcash_num_uncompromised} = stats || {};

    return (
        <div className="query-info col-md-12 col-lg-6">
            <div className="panel-sub">about your input</div>
            {!loading && <div className="panel-title">
                OVERALL INFO ON THE {amount} {currency} TORNADO CASH POOL
            </div>}


            {!loading && <div className="anon-score-group">
                uncompromised equal user deposits: {tcash_num_uncompromised} / {num_deposits}
            </div>}
            {loading && <div id="spinner" className="justify-content-center">
                <div className="spinner-border" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            </div>}
        </div>
    );
}