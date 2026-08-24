# Hunting Leopards with Half a Brain

Speaking script, single presenter, 5 minutes.

Each section has one job, and each one ends by handing off to the next.
1 who we are, 2 what we built, 3 why this animal, 4 the architecture, 5 the data and the cut,
6 what crosses the link, 7 what it cost and what it gained, 8 the consequence.

The abstract is the technical summary only. The species and the off-grid constraint belong to the
introduction, and the introduction is what sets up the architecture.

Pace note: if you are running behind, the sentence to drop is the encoder mechanism on slide 6.
The two numbers after it are what matter, not the channel counts.

---

## Slide 1: Title (0:15)

Good morning. I am happy to present our project, Hunting Leopards with Half a Brain, where we built
a split detection system for the Arabian leopard that runs on low powered hardware and does not rely
on the cloud.

---

## Slide 2: Abstract (0:39)

Here is the whole idea on one slide, and then I will unpack it.

We take a single YOLO26 Large model and cut it in two. The first three layers run on a small camera
node in the field, everything after them on one shared server nearby. Right at the cut we put a
learned compression step, and that is the part that makes the whole thing practical. What comes out
is a large model running on hardware that could never hold it, for almost no accuracy cost.

But before the how, here is why we are pointing it at this particular animal.

---

## Slide 3: Introduction (0:41)

About two hundred Arabian leopards remain in the wild, and the International Union for Conservation
of Nature lists the subspecies as critically endangered.

Saudi Arabia is putting serious money behind bringing it back. The Royal Commission for AlUla runs
the Arabian Leopard Fund, the breeding centre in Taif produced six cubs last year, and prey species
are being reintroduced to rebuild the food chain.

All of it depends on one input: knowing where the animals actually are, right now. And the leopard
lives in mountains with no power grid and no network, so whatever collects that has to survive out
there on its own. That constraint is what we have to work with

---

## Slide 4: System Overview (0:41)

So here is the shape of it.

The camera node is a Raspberry Pi Zero 2 W. It runs the first three layers of the model and nothing
else. What comes out goes over local Wi-Fi to a Raspberry Pi 3, which runs the remaining twenty
layers and the detection head.

It is arranged this way because of scale. The heavy part of the model exists once and is shared, so
covering more ground costs one more camera, not one more server.

And if the link drops, the camera does not go dark. It falls back to a smaller model it runs alone
until the link returns.

---

## Slide 5: Methodology, the data and the cut (0:43)

Two decisions sit behind that picture: what the model learns from, and where exactly to cut it.

The training set came from LILA BC, a public camera trap archive, plus seven open Roboflow datasets
and our own background photographs. Eight classes are labelled, but the leopard is the only one we
act on. The other seven share its habitat, and they teach the model what a leopard is not.

The cut point we did not guess. We enumerated the eighteen places in the graph where exactly one
tensor crosses the boundary, and profiled all of them on the devices. Layer three won, leaving three
point four percent of the parameters on the camera.

---

## Slide 6: Methodology, the bottleneck (0:50)

But where you cut says nothing about what crosses the link, and that is what decides whether this
scales.

At layer three the tensor you have to send is sixteen hundred kilobytes per frame. On one link
shared by many cameras, that alone caps how many nodes a server can carry.

So we trained an encoder and decoder pair that straddles the cut. The encoder narrows the feature
map from two hundred and fifty six channels down to sixteen, sent as eight bit integers. It costs
four thousand parameters on the camera, and the expensive reconstruction sits on the server.

Two numbers come out of that. The payload drops sixty four times, to twenty five kilobytes. And
transmission time drops seven times, from five hundred and twenty two milliseconds to seventy three,
measured on the devices.

---

## Slide 7: Results and Discussion (0:41)

Compression that aggressive usually costs you something, so the fair question is what it cost.

Almost nothing where it matters. On the leopard the model reaches nought point nine eight eight
average precision, and the compression takes about one hundredth of a point off that. Across all
eight classes together the drop is two hundredths. These are measured on the deployed ONNX artefacts
over a real link, not simulated.

On speed we actually gained. Overlapping the camera and the server so two frames are in flight at
once takes us from nought point three two to nought point five frames per second, fifty eight
percent faster on identical hardware.

---

## Slide 8: Conclusion (0:23)

Which brings me back to where I started.

A model that no single field device can run in real time now runs across a whole reserve, and the
wireless link is no longer what limits how many cameras that reserve can carry.

Next steps are skipping empty frames on the camera itself, and a live trial in the field.

Thank you.
