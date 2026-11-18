import { useEffect, useState } from 'react';
import Nav from './components/Nav.jsx';
import Home from './views/Home.jsx';
import News from './views/News.jsx';
import Audio from './views/Audio.jsx';
import { TITLES } from './lib/data.js';
import { reducedMotion } from './lib/utils.js';

const ROUTES = ['home', 'news', 'audio'];
const routeFromHash = () => {
  const h = window.location.hash.replace('#/', '');
  return ROUTES.includes(h) ? h : 'home';
};

function useHashRoute() {
  const [route, setRoute] = useState(routeFromHash);
  useEffect(() => {
    const on = () => setRoute(routeFromHash());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export default function App() {
  const route = useHashRoute();
  const [shown, setShown] = useState(route);
  const [leaving, setLeaving] = useState(false);

  // fade the old view out, then show the new one
  useEffect(() => {
    if (route === shown) return undefined;
    const show = () => { setLeaving(false); setShown(route); window.scrollTo(0, 0); };
    if (reducedMotion()) { show(); return undefined; }
    setLeaving(true);
    const t = setTimeout(show, 200);
    return () => clearTimeout(t);
  }, [route, shown]);

  useEffect(() => { document.title = TITLES[shown]; }, [shown]);

  // All views stay mounted so their state survives tab switches, like the original.
  const cls = (name) => 'view' + (shown === name ? ' on' : '') + (shown === name && leaving ? ' leave' : '');

  return (
    <>
      <Nav route={route} />
      <main>
        <section className={cls('home')} aria-label="Home"><Home /></section>
        <section className={cls('news')} aria-label="Fake news detection"><News active={shown === 'news'} /></section>
        <section className={cls('audio')} aria-label="Audio deepfake detection"><Audio active={shown === 'audio'} /></section>
      </main>
    </>
  );
}
