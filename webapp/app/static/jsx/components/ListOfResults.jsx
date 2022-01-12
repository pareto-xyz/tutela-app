import React, { useState } from 'react';
import { ListGroup, Toast, } from 'react-bootstrap';
import { CopyToClipboard } from 'react-copy-to-clipboard';

const Spinner = (<div id="spinner" className="justify-content-center">
    <div className="spinner-border" role="status">
        <span className="sr-only">Loading...</span>
    </div>
</div>)

export default function ListOfResults(props) {
    const { results, loading, aliases, rowTitle, rowBadge, sectionHeader,
        noDataComponent, } = props;
    const noResults = results.length == 0;

    function Row({ result, idx }) {
        const title = result[rowTitle];
        let badge = result[rowBadge];
        if (aliases[badge]) {
            badge = aliases[badge];
        }

        const [showToast, setShowToast] = useState(false);

        const [selected, setSelected] = useState(false);
        return (
            <div className={`row ${selected && 'selected-result'}`} key={idx}>
                <div className="col-12">
                    <ListGroup.Item className="my-accordion-header" onClick={() => setSelected(!selected)}>
                        <div className="row adress-row">
                            <div className="col-8 col-sm-5 col-md-4 col-lg-3 list-badge">{badge}</div>
                            <div className="col-md-12 col-lg-9 list-text">{title}</div>

                        </div>
                        <div className="row">
                            <div className="col-1 list-copy">
                                <CopyToClipboard text={title} onCopy={() => setShowToast(true)}><i className="far fa-copy"></i></CopyToClipboard>
                                <Toast className="copied-badge" onClose={() => setShowToast(false)} show={showToast} delay={3000} autohide>
                                    Copied!
                                </Toast>
                            </div>
                        </div>
                    </ListGroup.Item>
                </div>
            </div>
        );
    }

    return (
        <div className="col-12">
            <div className="row">
                {sectionHeader && sectionHeader}
                {(loading || noResults) && <div className="results loading col-12">

                    {loading ? Spinner : (noResults && noDataComponent)
                    }
                </div>}
                <ListGroup className="col-12" >
                    {results.map((result, idx) => <Row key={idx} result={result} idx={idx}></Row>)}
                </ListGroup>
                <div className="results loading col-12">
                    {loading && Spinner}
                </div>
            </div>
        </div>
    )
}