import React from 'react';

export default function TpoolOverall({data, loading}) {
    const {metadata} = data;
    const {amount, currency, stats} = metadata || {};
    const {num_deposits, num_uncompromised} = stats || {};

    return (
        <div className="query-info ">
            <div className="panel-sub">about your input</div>
            {!loading && <div className="panel-title">
                OVERALL INFO ON THE {amount} {currency} TORNADO CASH POOL
            </div>}


            {!loading && <div className="anon-score-group">
                uncompromised equal user deposits: {num_uncompromised} / {num_deposits}
            </div>}
            {loading && <div id="spinner" className="justify-content-center">
                <div className="spinner-border" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            </div>}
        </div>
    );
}