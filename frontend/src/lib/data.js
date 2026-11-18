export const NASTALIQ_SAMPLE = 'حکومت نے اعلان کیا ہے کہ اگلے ماہ سے تمام سرکاری دفاتر ہفتے میں صرف تین دن کھلیں گے۔';
export const ROMAN_SAMPLE = 'Hukoomat ne elaan kiya hai ke agle mahine se tamam sarkari daftar hafte mein sirf teen din khulenge.';
export const ROMAN_TRANSLIT = 'حکومت نے اعلان کیا ہے کہ اگلے مہینے سے تمام سرکاری دفاتر ہفتے میں صرف تین دن کھلیں گے۔';

export const TICKER = [
  ['Politics', 'سیاست'], ['Business', 'کاروبار'], ['Health', 'صحت'], ['Sports', 'کھیل'],
  ['Technology', 'ٹیکنالوجی'], ['Entertainment', 'تفریح'], ['Education', 'تعلیم'], ['General', 'عمومی'],
  ['Nastaliq', 'نستعلیق'], ['Roman Urdu', 'رومن اردو'], ['Screenshots', 'اسکرین شاٹس'],
  ['Voice clones', 'نقلی آوازیں'], ['Text to speech', 'مصنوعی گفتار']
];

export const DOMAINS = [
  ['politics', 'Politics', 'سیاست'], ['business', 'Business', 'کاروبار'], ['health', 'Health', 'صحت'],
  ['sports', 'Sports', 'کھیل'], ['tech', 'Technology', 'ٹیکنالوجی'], ['entertainment', 'Entertainment', 'تفریح'],
  ['education', 'Education', 'تعلیم'], ['general', 'General', 'عمومی']
];

export const NEWS_OUT = {
  fake: {
    cls: 'bad', en: 'Likely fake', ur: 'غالباً جعلی', w: 82,
    why: [
      'No official notification, ministry or spokesperson is cited for a nationwide policy change.',
      'The claim is sweeping and specific at once, a pattern common in forwarded rumours.',
      'Phrasing resembles items labelled fake in the training corpora for this domain.'
    ]
  },
  real: {
    cls: 'good', en: 'Likely real', ur: 'غالباً مستند', w: 78,
    why: [
      'A named ministry and a dated notification are cited.',
      'Details are specific and checkable, in the register of official reporting.',
      'Phrasing resembles items labelled real in the training corpora for this domain.'
    ]
  }
};

export const AUDIO_OUT = {
  synthetic: {
    cls: 'bad', en: 'Likely synthetic', ur: 'غالباً مصنوعی آواز', w: 76,
    why: [
      'Pitch contour stays unusually flat across sentence boundaries.',
      'Breath and mouth noise are missing between phrases in the flagged segments.',
      'Spectral artefacts match patterns the model learned from TTS voices.'
    ]
  },
  genuine: {
    cls: 'good', en: 'Likely genuine', ur: 'غالباً اصلی آواز', w: 18,
    why: [
      'Pitch varies naturally across sentences and stressed syllables.',
      'Breath and room noise are present throughout the clip.',
      'No spectral artefacts typical of TTS voices were found.'
    ]
  }
};

export const TITLES = {
  home: 'Sachai Check · Urdu news and voice verification',
  news: 'Fake News Detection · Sachai Check',
  audio: 'Audio Deepfake Detection · Sachai Check'
};
